import torch
import math
from torch import nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3, resnet18

class DCNBlock(nn.Module):
    def __init__(self, in_planes, planes,k,p,offset = 1,d = 1,offgrad = True):
        super(DCNBlock, self).__init__()
        self.kernel = k
        self.channel = in_planes
        self.offset = offset
            # k = off*2 +1,padding = off*d
        self.offconv = nn.Conv2d(in_planes, 2 * in_planes , kernel_size= (offset *2 + 1),
                stride=1, dilation = d, padding= (offset * d))
        for par in self.parameters():
            par.requires_grad = offgrad
            self.dcnconv = nn.Conv2d(in_planes, planes, kernel_size=k,
                stride=1,dilation=d, padding=p)
    def forward(self, x):
        B,C,H,W = x.shape
        off = (torch.sigmoid(self.offconv(x)) - 0.5) * 2 * self.offset
        off = off.reshape((B*C,2,H,W))
        off[:,0,:,:] /= (W - 1)
        off[:,1,:,:] /= (H - 1)
        grid = off.permute(0,2,3,1).clone()
        X,Y = torch.meshgrid(torch.arange(0, H,dtype = torch.float32,device=x.device), torch.arange(0, W,dtype = torch.float32,device=x.device))
        X = (X/(H - 1)) * 2 - 1
        Y = (Y/(W - 1)) * 2 - 1
        grid[:,:,:,0] = Y.expand(B*C,H,W)
        grid[:,:,:,1] = X.expand(B*C,H,W)
        grid = grid - off.permute(0,2,3,1)
        x = x.reshape((B*C,1,H,W))
        out = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection',align_corners=True)
        out = out.reshape((B,C,H,W))
        out = self.dcnconv(out)
        return out

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model = 32, max_shape=(128, 128)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
       # print(x.shape)
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
   #     Q = Q * q_mask[:, :, None,None]
   #     if kv_mask is not None:
   #         K = K * kv_mask[:, :, None, None]
 #           values = values * kv_mask[:, :, None, None]
        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        return queried_values.contiguous()

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model = 32,
                 nhead = 8,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x,source):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        B,C,H,W = x.shape
        x = x.reshape((B,C,H*W)).permute(0,2,1)
        source = source.reshape((B,C,H*W)).permute(0,2,1)
        bs = B
        query, key, value = x, source, source
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)
        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        x = x + message
        x = x.permute(0,2,1).reshape((B,C,H,W))
        return x

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self):
        super(LocalFeatureTransformer, self).__init__()

       # self.config = config
       # self.d_model = config['d_model']
       # self.nhead = config['nhead']
      #  self.layer_names = config['layer_names']
        self.encoder_layer = LoFTREncoderLayer()
      #  self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat1,feat2):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        "the feature number of src and transformer must be equal"

       # for layer, name in zip(self.layers, self.layer_names):
        feat = self.encoder_layer(feat1,feat2)
        return feat

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, (channel // reduction)**2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((channel // reduction)**2, channel**2, bias=False),
            nn.Sigmoid() 
        )
        self.dc = nn.Sequential(
        #    nn.Conv2d(1, channel // reduction, kernel_size=(1,1)),
         #   DCNBlock(1, channel // reduction,k = 3,p =1 , offset = 1,d = 1,offgrad = True),
       #     nn.BatchNorm2d(channel // reduction,affine = False),
       #     nn.ReLU(inplace=True),
        #    nn.Conv2d(channel, channel, kernel_size=(3,3), stride=(1,1),dilation=(1,1), padding=(1,1)),
       #     nn.Conv2d(channel // reduction, channel, kernel_size=(1,1)),
           # nn.MaxPool2d(3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x,depth,depth_ori):
        '''
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, c)
        y = F.normalize(y,p = 2,dim = 1)
        x = x.view(b,c,h*w)
        x = y.bmm(x).view(b,c,h,w)
        '''
#        print(depth.shape)
        b,c,h,w = x.shape
        b,_,h0,w0 = depth.shape
        grid = 3
 #       print(x.shape)
        d = self.dc(depth)
        d = F.interpolate(d, size=[h, w], mode="bilinear")
        d = d.reshape((b,grid,4*c,h,w))
        rd = torch.arange(0,grid,device = x.device).reshape((1,grid,1,1))
        rd = (1 - torch.abs(rd - depth_ori*grid))
        rd[rd < 0] = 0
        rd = rd.unsqueeze(2)
#        print(d.shape)
        d = d*rd
        d = d.sum(axis = 1)
      #  print(d.shape)
        d = d.reshape((b,4,c,h,w))
        eye = torch.eye(4,device = d.device).reshape((1,4,4,1,1,1))
   #     mat = d[:,0:4,:,:,:]
      #  bias = d[:,4,:,:,:].reshape((b,c,h,w))
#       mat = mat.reshape((b,4,1,c // 4,h,w))
        mat = d.reshape((b,4,4,c // 4,h,w))
        mat = mat + eye
#       x = torch.sum((mat*x),axis = 1)
        x = x.reshape((b,1,4,c //4,h,w))
        x = torch.sum((mat*x),axis = 2)
        x = x.reshape((b,c,h,w))
       # x = x + bias
#        print(d.shape)
    
        return x
'''
class SEBlock(nn.Module):
    def __init__(self, in_planes, r = 2):
        super(SEBlock,self).__init__()
        r_planes = int(in_planes / r)
        self.fc1 =  nn.Linear(in_planes,r_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 =  nn.Linear(r_planes,in_planes)
    def forward(self,x):
        x = torch.mean(x,dim = (2,3))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.unsqueeze(2).unsqueeze(3)
       # print(x.shape)
        return x
'''


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_c, out_c, batchnorm=True, activation=True, k=3):
        super().__init__()
        if k == 3:
            self.conv = conv3x3(in_c, out_c)
        elif k == 1:
            self.conv = conv1x1(in_c, out_c)
        else:
            raise ValueError()

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_c)
        else:
            self.bn = nn.Identity()

        if activation:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_c, out_c, upsampling_method):
        super().__init__()
        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1),
            )

    def forward(self, x):
        return self.upsample(x)


class DCNEncoder(nn.Module):
    def __init__(self, chan_in=3, chan_out=32, pretrained=True):
        super().__init__()
        self.chan_out = chan_out
        self.inconv = ConvBlock(chan_in, 64, k=3)
        dexpand = 4
        grid = 3
        p = 0.2
        self.gconv = nn.Sequential(
                nn.Conv2d(1,3,kernel_size = (3,3),padding = 1),
                nn.BatchNorm2d(3,affine = False),
                nn.ReLU(inplace=True),
               # nn.Dropout(p = p),
                nn.Conv2d(3,1,kernel_size = (3,3),padding = 1),
                nn.Sigmoid()
                )
      #  nn.Conv2d(4,1,kernel_size = )
        self.dconv = nn.Sequential(
                nn.Conv2d(1, 8*dexpand, kernel_size=(3,3),stride = 2,padding = 1),
                #   DCNBlock(1, channel // reduction,k = 3,p =1 , offset = 1,d = 1,offgrad = True),
                nn.BatchNorm2d(8*dexpand,affine = False),
                nn.ReLU(inplace=True),
              #  nn.Dropout(p = p),
                #    nn.Conv2d(channel, channel, kernel_size=(3,3), stride=(1,1),dilation=(1,1), padding=(1,1)),
                nn.Conv2d(8*dexpand, 64*dexpand, kernel_size=(3,3),stride = 2,padding = 1),
                nn.BatchNorm2d(64*dexpand,affine = False),
                nn.ReLU(inplace=True),
              #  nn.Dropout(p = p),
                #    nn.Conv2d(channel, channel, kernel_size=(3,3), stride=(1,1),dilation=(1,1), padding=(1,1)),
                nn.Conv2d(64*dexpand, 64*dexpand*grid, kernel_size=(3,3),stride = 2,padding = 1),
                #g nn.MaxPool2d(3, stride=1, padding=1),
               # nn.Sigmoid()
                )
        self.layer1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),dilation=(1,1), padding=(1,1)),
              #  DCNBlock(64, 64,k = 3,p =1 , offset = 1,d = 1,offgrad = True),
                nn.BatchNorm2d(64,affine = False),
                nn.ReLU(inplace=True),
              #  nn.Dropout(p = p),
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),dilation=(1,1), padding=(1,1)),
                nn.MaxPool2d(3, stride=1, padding=1),
                )
        self.dlayer1 = nn.Sequential(
                nn.BatchNorm2d(64*dexpand*grid,affine = False),
                nn.ReLU(inplace=True),
              #  nn.Dropout(p = p),
                nn.Conv2d(64*dexpand*grid, 64*dexpand*grid, kernel_size=(1,1)),
                )
        self.se1 = SEBlock(64)
        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),dilation=(2,2), padding=(2,2)),
               # DCNBlock(64, 64,k = 3,p =2 , offset = 1,d = 2,offgrad = True),
                nn.BatchNorm2d(64,affine = False),
                nn.ReLU(inplace=True),
               # nn.Dropout(p = p),
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),dilation=(2,2), padding=(2,2)),
                nn.MaxPool2d(3, stride=1, padding=1),
                )
        self.dlayer2 = nn.Sequential(
                nn.BatchNorm2d(64*dexpand*grid,affine = False),
                nn.ReLU(inplace=True),
               # nn.Dropout(p = p),
                nn.Conv2d(64*dexpand*grid, 64*dexpand*grid, kernel_size=(1,1)),
                )
        self.se2 = SEBlock(64)
        self.layer3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),dilation=(4,4), padding=(4,4)),
               # DCNBlock(64, 64,k = 3,p =4 , offset = 1,d = 4,offgrad = True),
                nn.BatchNorm2d(64,affine = False),
                nn.ReLU(inplace=True),
               # nn.Dropout(p = p),
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),dilation=(4,4), padding=(4,4)),
                nn.MaxPool2d(3, stride=1, padding=1),
                )
        self.dlayer3 = nn.Sequential(
                nn.BatchNorm2d(64*dexpand*grid,affine = False),
                nn.ReLU(inplace=True),
              #  nn.Dropout(p = p),
                nn.Conv2d(64*dexpand*grid, 64*dexpand*grid,kernel_size=(1,1)),
                )
        self.se3 = SEBlock(64)
        self.layer4 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),dilation=(8,8), padding=(8,8)),
               # DCNBlock(64, 64,k = 3,p =8 , offset = 1,d = 8,offgrad = True),
                nn.BatchNorm2d(64,affine = False),
                nn.ReLU(inplace=True),
              #  nn.Dropout(p = p),
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),dilation=(8,8), padding=(8,8)),
                nn.MaxPool2d(3, stride=1, padding=1),
                )
        self.dlayer4 = nn.Sequential(
                nn.BatchNorm2d(64*dexpand*grid,affine = False),
                nn.ReLU(inplace=True),
               # nn.Dropout(p = p),
                nn.Conv2d(64*dexpand*grid, 64*dexpand*grid, kernel_size=(1,1)),
                )
        self.se4 = SEBlock(64)
      #  self.layer1 = resnet.layer1
 #       self.layer2 = resnet.layer1
        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)
        self.outconv2 = ConvBlock(64, chan_out, k=1, activation=False)
        self.doutconv = nn.Sequential(
                            ConvBlock(64, chan_out, k=1, activation=False),
                            nn.Sigmoid()
                        )
        self.dencoder = nn.Sequential(
                            ConvBlock(1, 8, k=3),
                            DCNBlock(8, 8,k = 3,p =1 , offset = 1,d = 1,offgrad = True),
                            nn.BatchNorm2d(8,affine = False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(8, 8, kernel_size=(3,3), stride=(1,1),dilation=(1,1), padding=(1,1)),
                            nn.MaxPool2d(3, stride=1, padding=1),
                            ConvBlock(8, 64, k=3)
                        )
       # self.posencoder = PositionEncodingSine()
       # self.transformer = LocalFeatureTransformer()
    def forward(self, x,d):
        # first apply the resnet blocks up to layer 2
       # print("encoder")
    #    x = torch.cat((x,d),1)
   #     guide = torch.cat((x,d),1)
        g = self.gconv(d)
        x = self.inconv(x)
       # g = self.gconv(x)
        d0 = self.dconv(d)
        x1 = self.layer1(x)   # -> 64 x H/2 x W/2
        d1 = self.dlayer1(d0)
       # print(x.shape)
        x2 = self.layer2(x1)  # -> 64 x H/2 x W/2
        d2 = self.dlayer2(d1)
        x = (self.se1(x1,d1,g) + self.se2(x2,d2,g)) * 0.5
     #   x3 = self.layer3(x2)
     #   d3 = self.dlayer(d2)
     #   x4 = self.layer4(x3)
     #   d4 = self.dlayer(d3)
       # print(x.shape)
     #   x = (self.se1(x1,d1) + self.se2(x2,d2) + self.se3(x3,d3) + self.se4(x4,d4)) * 0.25
     #   b,c,h,w = x.shape
        x = self.outconv(x)
     #   b,c,h,w = x.shape
     #   x = x.view(b,8,c // 8,h,w)
     #   rd = torch.arange(0,8,device = x.device).reshape((1,8,1,1))
     #   rd = (1 - torch.abs(rd - d))
     #   rd[rd < 0] = 0
     #   print(rd)

       # xb = self.outconv2(x)
       # dd = self.doutconv(d2)
      #  x = d*xa +(1 - d)*xb
    #    x = x[:,0:self.chan_out,:,:] * d + (1 - x[:,self.chan_out:self.chan_out*2,:,:])*(1 - d)
       # x = self.posencoder(x)
       # x = self.transformer(x,x)
    #    d = self.dencoder(d)
     #   x = x*d
       # import torch
       # x = self.outconv(x)
   #     x = F.normalize(x,p = 2,dim = 1)
       # print(torch.sum(x**2,axis = 1))
       # x = x / (torch.sqrt(torch.sum(x**2,axis = 1)).unsqueeze(1) + 1e-16)
        return x


class DCNDecoder(nn.Module):
    def __init__(self, chan_in, chan_out, non_linearity, pretrained=False):
        super().__init__()
        self.outconv = ConvBlock(chan_in, chan_out, batchnorm=False, activation=False)
        if non_linearity is None:
            self.non_linearity = nn.Identity()
        else:
            self.non_linearity = non_linearity

    def forward(self, x):
        x = self.outconv(x)  # -> C_out x H x W
        x = self.non_linearity(x)
        return x

class DepthEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.patches = nn.Unfold(5, padding=2, stride=1)
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.inconv = ConvBlock(1, 3, batchnorm=False, activation=False)
        self.layer = nn.Sequential(
                DCNBlock(3, 9,k = 3,p =1 , offset = 1,d = 1,offgrad = True),
                nn.BatchNorm2d(9,affine = False),
                nn.ReLU(inplace=True),
                nn.Conv2d(9, 3, kernel_size=(3,3), stride=(1,1),dilation=(1,1), padding=(1,1)),
                nn.MaxPool2d(3, stride=1, padding=1),
                )
        self.outconv = ConvBlock(3, 1, batchnorm=False, activation=False)
    def JBF(self,x,gray):
        B,_,H,W = x.shape
        X,Y = torch.meshgrid(torch.arange(0, H,dtype = torch.float32,device=x.device), torch.arange(0, W,dtype = torch.float32,device=x.device))
        xb = self.patches(x)
        gb = self.patches(gray)
        Xb = self.patches(X.unsqueeze(0).unsqueeze(1))
        Yb = self.patches(Y.unsqueeze(0).unsqueeze(1))
     #   x0 = x.reshape((B,1,H*W))
        X0 = X.reshape((1,1,H*W))
        Y0 = Y.reshape((1,1,H*W))
        g0 = gray.reshape((B,1,H*W))
        xf = torch.exp(- torch.sqrt(((Xb - X0)**2 + (Yb - Y0)**2))/8)
        xg = torch.exp(-(torch.abs(gb - g0))/8)
        J = xb*xf*xg
       # print(Xb[:,:,10000])
        #print(X0[:,:,10000])
        J = torch.mean(J,axis = 1)
        J = J.reshape((B,1,H,W))
       # print(J)
       # print(torch.sqrt(((Xb - X0)**2 + (Yb - Y0)**2))*xb)
       # print(x0.shape)
       # print(X0.shape)
       # print(Y0.shape)
      #  print(X)
       # print(Y)
        return J
    def forward(self, x,rgb):
   #     print(x.shape)
        gray = torch.mean(rgb,axis = 1).unsqueeze(1)
        B,_,H,W = x.shape
        eps = 1e-16
        x_1 = (x != 0).float()
       # x_sq = x.reshape((B,1,H*W))
       # x_mask = (x_sq != 0).int()
       # x_mean = (x_sq.sum(axis = 2) / (x_mask.sum(axis = 2) + 1e-16)).unsqueeze(2)
       # x_std = torch.sqrt(torch.sum((x_sq - x_mean)**2 * x_mask, axis =2) / (x_mask.sum(axis = 2) + 1e-16)).unsqueeze(2)
       # x_mean = x_mean.unsqueeze(3)
       # x_std = x_std.unsqueeze(3)
  #      x = ((x - x_mean) / (3 * x_std + 1e-16))
        x = torch.exp(x) * x_1 / (torch.exp(x) + 1)
      #  print("start")
     #   print(x)
        while x_1.min() < (1.0 -  eps):
            x0 = self.JBF(x,gray)
            xm = self.JBF(x_1,gray)
           # x0 = F.conv2d(x, self.weight, padding=2, groups=1)
           # xm = F.conv2d(x_1,self.weight,padding = 2,groups = 1)
        #    print(x_1)
     #       print(x0)
     #       print(xm)
            x[x_1 == 0] = (x0[x_1 ==0] / (xm[x_1 == 0] + eps))
            x_1 = (x != 0).float()
         #   print(x)
          #  x_mask = (x.reshape((B,1,H*W)) != 0).int()
           # print(x_1.mean())
    #    print((x - xori))
  #      print(x)
       # x_sq = x.reshape((B,1,H*W))
       # x_max = x_sq.max(axis = 2)[0].unsqueeze(2).unsqueeze(3)
       # x_min = x_sq.min(axis = 2)[0].unsqueeze(2).unsqueeze(3)
        #x = (x - x_min) / (x_max - x_min + 1e-16)
       # x = x / (nn.MaxPool2d((H,W))(x) + 1e-16)
       # x = self.inconv(x)  # -> C_out x H x W
       # x = self.layer(x)
       # x = self.outconv(x)
       # x = x**2
     #   x = torch.exp(x) * x_1 / (torch.exp(x) + 1)
        x = (x - 0.5) *2
        return x

class DCNMask(nn.Module):
    def __init__(self, chan_in, pretrained=False):
         super().__init__()
         self.outconv = ConvBlock(chan_in, 1,batchnorm=False, activation=False,k=1)
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]
    def forward(self, x):
        x = self.outconv(x)
        x = self.softmax(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, chan_in, chan_out, non_linearity, pretrained=False):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        resnet.inplanes = chan_in
        self.layer1 = resnet._make_layer(BasicBlock, 64, 2)
        resnet.inplanes = 64
        self.layer2 = resnet._make_layer(BasicBlock, 64, 2)

        self.upconv1 = UpConv(64, 64, "bilinear")
        self.outconv = ConvBlock(64, chan_out, batchnorm=False, activation=False)

        if non_linearity is None:
            self.non_linearity = nn.Identity()
        else:
            self.non_linearity = non_linearity

        # Initialize all the new layers
        self.resnet_init()

    def resnet_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.layer1(x)  # -> 128 x H/4 x W/4
        x = self.layer2(x)  # -> 64 x H/2 x W/2
        x = self.outconv(x)  # -> C_out x H x W
        x = self.non_linearity(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.inconv = ConvBlock(chan_in, 64, k=3)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer1
        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x = self.layer1(x)  # -> 64 x H/2 x W/2
        x = self.layer2(x)  # -> 64 x H/2 x W/2
        x = self.outconv(x)

        return x

