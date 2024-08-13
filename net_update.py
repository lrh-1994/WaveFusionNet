import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import numpy as np
from fusion_block import SwinFusion
EPSILON = 1e-10

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.batchnormal = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.activate = nn.Tanh()
        #self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is True:
            #out = self.batchnormal(out)
            out = self.lrelu(out)
            # out = self.dropout(out)
        return out


class Multi_scale_res_block(nn.Module):
    def __init__(self):
        super(Multi_scale_res_block, self).__init__()
       
        self.conv11 = ConvLayer(32, 32, 1, 1,is_last=False)
        self.conv13 = ConvLayer(32, 32, 3, 1,is_last=False)
        self.conv15 = ConvLayer(32, 32, 5, 1,is_last=False)

        self.conv21 = ConvLayer(32, 32, 1, 1)
        self.conv23 = ConvLayer(32, 32, 3, 1)

        self.convbottle = ConvLayer(64, 32, 1, 1, is_last=False)
        #self.conv52 = ConvLayer(64, 32, 5, 1)
        #self.conv53 = ConvLayer(96, 32, 5, 1)
        #self.convtrue = ConvLayer(96, 32, 1, 1, is_last=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        conv11 = self.conv11(x)
       
        conv13 = self.conv13(x)

        conv21 = self.conv21(torch.cat([conv11,conv13],1))
        
        conv23 = self.conv23(torch.cat([conv11,conv13],1))

        convbottle = self.convbottle(torch.cat([conv21,conv23],1))
        #convbpttle = convbottle + x
        out=self.sig(convbottle)
        
        return out
        
class network_vr(nn.Module):
    def __init__(self):
        super(network_vr, self).__init__()
        self.vi_encoder=network_vi_encoder()
        self.ir_encoder=network_ir_encoder()
        self.decoder=network_decoder()
        
    def forward(self,vi,ir,dwt,idwt):
    
        vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h=self.vi_encoder(vi,dwt)	
        ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h=self.ir_encoder(ir,dwt)
        
        vi_out=self.decoder(vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h,idwt)
        ir_out=self.decoder(ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h,idwt)
        
        return vi_out,ir_out

        
class network_fusion(nn.Module):
    def __init__(self):
        super(network_fusion, self).__init__()
        self.feature_fusion =feature_fusion()
        self.vr=network_vr()
    def forward(self,vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l):
        #f64_l,f32_l,f16_l = self.feature_fusion(vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l)
        return 0#f64_l,f32_l,f16_l   
         
class feature_fusion(nn.Module):
    def __init__(self):
        super(feature_fusion, self).__init__()

        self.fusion1 = SwinFusion(img_size=64)
        self.fusion1_conv = ConvFusion()
        self.fusion2 = SwinFusion(img_size=32)
        self.fusion2_conv = ConvFusion()
        self.fusion3 = SwinFusion(img_size=16)
        self.fusion3_conv = ConvFusion()
        self.m_pool=torch.nn.AvgPool2d(2,2)
        self.f_pool=torch.nn.AvgPool2d(2,2)
        
    def forward(self,vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l):

        #f64_l_t =self.fusion1(vi64_l,ir64_l)
        
        f64_l = self.fusion1_conv(vi64_l,ir64_l)

        #f32_l_t =self.fusion2(vi32_l,ir32_l)
        
        f32_l = self.fusion2_conv(vi32_l,ir32_l)

        #f16_l_t =self.fusion3(vi16_l,ir16_l)
        
        f16_l = self.fusion3_conv(vi16_l,ir16_l)
        
        #f_pool=self.f_pool(f64_l_c)
        #m_pool=self.m_pool(torch.max(vi64_l,ir64_l))
        #f64_l= f64_l_t + f64_l_c
        #f32_l= f32_l_t + f32_l_c
        #f16_l= f16_l_t + f16_l_c
        
        return f64_l,f32_l,f16_l#,f64_l_g,f32_l_g,f16_l_g#f_pool,m_pool#f64_l,f32_l,f16_l,f64_l_t,f32_l_t,f16_l_t,

class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 32, 1)

    def forward(self, x0, x1):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x1), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x1), 0.1, inplace=True))
        return x0 * (scale + 1) + shift


class SKFF_Fusion(nn.Module):
    def __init__(self):
        super(SKFF_Fusion,self).__init__()

        self.msb_vi = ContextBlock(32)
        self.msb_ir = ContextBlock(32)
        
        self.skff = SKFF(32,2,8,bias=False)
 
    def forward(self,vi,ir):
    
        vi=self.msb_vi(vi)
        ir=self.msb_ir(ir)
        
        out=self.skff([vi,ir])
        
        return out

class network_vi_encoder(nn.Module):
    def __init__(self):
        super(network_vi_encoder,self).__init__()
        
        self.conv_vi1 = ConvLayer(1, 32, 3, 1,is_last=False)
        self.relu1=nn.ReLU()#LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.conv_vi2 = ConvLayer(32, 32, 3, 1,is_last=False)
        self.relu2=nn.ReLU()#LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.conv_vi3 = ConvLayer(32, 32, 3, 1,is_last=False)
        self.relu3=nn.ReLU()#LeakyReLU(negative_slope=0.1, inplace=True)
        

    def forward(self, vi, dwt):
        #print(vi.size())
        vi1 =  self.conv_vi1(vi)
        vi1 =  self.relu1(vi1)

        vi64_l,vi64_h = dwt(vi1)
        vi64_cat = vi64_l#torch.cat([vi64_l,vi64_h[0][:,:,0,:,:],vi64_h[0][:,:,1,:,:],vi64_h[0][:,:,2,:,:]],1)
        
        vi2= self.conv_vi2(vi64_cat)
        vi2= self.relu2(vi2)
        
        vi32_l,vi32_h = dwt(vi2)
        vi32_cat = vi32_l#torch.cat([vi32_l,vi32_h[0][:,:,0,:,:],vi32_h[0][:,:,1,:,:],vi32_h[0][:,:,2,:,:]],1)
        
        vi3=self.conv_vi3(vi32_cat)
        vi3=self.relu3(vi3)
        
        vi16_l,vi16_h = dwt(vi3)
        #vi16_cat = vi16_l#torch.cat([vi16_l,vi16_h[0][:,:,0,:,:],vi16_h[0][:,:,1,:,:],vi16_h[0][:,:,2,:,:]],1)
        
        #vi6=self.conv_vi6(vi16_cat)
        #vi7=self.conv_vi7(vi6)
        #vi7=self.relu7(vi7)
        
        return vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h

class network_ir_encoder(nn.Module):
    def __init__(self):
        super(network_ir_encoder,self).__init__()
        
        self.conv_ir1 = ConvLayer(1, 32, 3, 1,is_last=False)
        self.relu1=nn.ReLU()#LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.conv_ir2 = ConvLayer(32, 32, 3, 1,is_last=False)
        self.relu2=nn.ReLU()#.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.conv_ir3 = ConvLayer(32, 32, 3, 1,is_last=False)
        self.relu3=nn.ReLU()#LeakyReLU(negative_slope=0.1, inplace=True)
        

    def forward(self, ir, dwt):
        #print(vi.size())
        ir1 =  self.conv_ir1(ir)
        ir1 =  self.relu1(ir1)
        
        ir64_l,ir64_h = dwt(ir1)
        ir64_cat = ir64_l#torch.cat([ir64_l,ir64_h[0][:,:,0,:,:],ir64_h[0][:,:,1,:,:],ir64_h[0][:,:,2,:,:]],1)
        
        ir2=self.conv_ir2(ir64_cat)
        ir2=self.relu2(ir2)
        
        ir32_l,ir32_h = dwt(ir2)
        ir32_cat = ir32_l#torch.cat([ir32_l,ir32_h[0][:,:,0,:,:],ir32_h[0][:,:,1,:,:],ir32_h[0][:,:,2,:,:]],1)
        
        ir3=self.conv_ir3(ir32_cat)
        ir3=self.relu3(ir3)
        
        ir16_l,ir16_h = dwt(ir3)
        #ir16_cat = ir16_l#torch.cat([ir16_l,ir16_h[0][:,:,0,:,:],ir16_h[0][:,:,1,:,:],ir16_h[0][:,:,2,:,:]],1)
        #ir6=self.conv_ir6(ir5)
        #ir7=self.conv_ir7(ir6)
        #ir7=self.relu7(ir7)
        
        return ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h

class network_decoder(nn.Module):
    def __init__(self):
        super(network_decoder, self).__init__()
        self.relu_last = nn.Tanh()
       
        self.msb161 = ContextBlock(32)
        
        self.msb162 = ContextBlock(32)
        
        self.msb321 = ContextBlock(32)
        
        self.msb322 = ContextBlock(32)
        
        self.msb641 = ContextBlock(32)
        
        self.msb642 = ContextBlock(32)

        self.sff321 = SKFF(32,2,8,bias=False)
        self.sff322 = SKFF(32,2,8,bias=False)
        self.sff641 = SKFF(32,2,8,bias=False)
        self.sff642 = SKFF(32,2,8,bias=False)
        
        self.conv_o = ConvLayer(32, 1, 3, 1,is_last=False)


    def forward(self,f64_l,f32_l,f16_l,f64_h,f32_h,f16_h,idwt):
    
        conv161 = self.msb161(f16_l)
        conv162 = self.msb162(conv161)

        conv321 = self.msb321(f32_l)
        conv32_cat1 = self.sff321([conv321,idwt((conv161,f16_h))])
        conv322 = self.msb322(conv32_cat1)
        conv32_cat2 = self.sff322([conv322,idwt((conv162,f16_h))])
        
        conv641 = self.msb641(f64_l)
        conv64_cat1 = self.sff641([conv641,idwt((conv32_cat1,f32_h))])
        conv642 = self.msb642(conv64_cat1)
        conv64_cat2 = self.sff642([conv642,idwt((conv32_cat2,f32_h))])

        result_image = self.conv_o(idwt((conv64_cat2,f64_h)))
        
        result_image = self.relu_last(result_image)
        
        return result_image

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V 

class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()
        self.conv_d = ConvLayer(32, 32, 3, 1,is_last=False)
        self.relu_d=nn.ReLU()#LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        #context = self.modeling(x)
        
        # [N, C, 1, 1]
        #channel_add_term = self.channel_add_conv(context)
        #x = x + channel_add_term
        x=self.conv_d(x)
        x=self.relu_d(x)
        return x


class ConvFusion_strategy1(nn.Module):
    def __init__(self):
        super(ConvFusion_strategy1,self).__init__()

        self.conv11_ir = ConvLayer(32, 32, 1, 1,is_last=False)
        self.conv13_ir = ConvLayer(32, 32, 3, 1,is_last=False)
        self.conv15_ir = ConvLayer(32, 32, 5, 1,is_last=False)

        self.conv21_ir = ConvLayer(96, 32, 1, 1,is_last=False)
        self.conv23_ir = ConvLayer(96, 32, 3, 1,is_last=False)
        self.conv25_ir = ConvLayer(96, 32, 5, 1,is_last=False)


        self.conv11_vi = ConvLayer(32, 32, 1, 1,is_last=False)
        self.conv13_vi = ConvLayer(32, 32, 3, 1,is_last=False)
        self.conv15_vi = ConvLayer(32, 32, 5, 1,is_last=False)

        self.conv21_vi = ConvLayer(96, 32, 1, 1,is_last=False)
        self.conv23_vi = ConvLayer(96, 32, 3, 1,is_last=False)
        self.conv25_vi = ConvLayer(96, 32, 5, 1,is_last=False)

        
        self.convall_1 = ConvLayer(192, 64, 1, 1,is_last=False)
        self.convall_2 = ConvLayer(64, 64, 1, 1,is_last=False)
        self.convall_3 = ConvLayer(128, 32, 1, 1,is_last=False)
        self.convall_4 = ConvLayer(160, 32, 1, 1,is_last=False)


        self.conv11_ir_relu=nn.ReLU()
        self.conv13_ir_relu=nn.ReLU()
        self.conv15_ir_relu=nn.ReLU()
        self.conv21_ir_relu=nn.ReLU()
        self.conv23_ir_relu=nn.ReLU()
        self.conv25_ir_relu=nn.ReLU()

        self.conv11_vi_relu=nn.ReLU()
        self.conv13_vi_relu=nn.ReLU()
        self.conv15_vi_relu=nn.ReLU()
        self.conv21_vi_relu=nn.ReLU()
        self.conv23_vi_relu=nn.ReLU()
        self.conv25_vi_relu=nn.ReLU()


        self.convall_1_relu=nn.ReLU()
        self.convall_2_relu=nn.ReLU()
        self.convall_3_relu=nn.ReLU()
        self.convall_4_relu=nn.ReLU()


    def forward(self, vi, ir):

        vi11=self.conv11_vi(vi)
        vi11=self.conv11_vi_relu(vi11)
        vi13=self.conv13_vi(vi)
        vi13=self.conv13_vi_relu(vi13)
        vi15=self.conv15_vi(vi)
        vi15=self.conv15_vi_relu(vi15)
        
        vi21=self.conv21_vi(torch.cat([vi11,vi13,vi15],1))
        vi21=self.conv21_vi_relu(vi21)
        vi23=self.conv23_vi(torch.cat([vi11,vi13,vi15],1))
        vi23=self.conv23_vi_relu(vi23)
        vi25=self.conv25_vi(torch.cat([vi11,vi13,vi15],1))
        vi25=self.conv25_vi_relu(vi25)

        ir11=self.conv11_ir(ir)
        ir11=self.conv11_ir_relu(ir11)
        ir13=self.conv13_ir(ir)
        ir13=self.conv13_ir_relu(ir13)
        ir15=self.conv15_ir(ir)
        ir15=self.conv15_ir_relu(ir15)
        
        ir21=self.conv21_ir(torch.cat([ir11,ir13,ir15],1))
        ir21=self.conv21_ir_relu(ir21)
        ir23=self.conv23_ir(torch.cat([ir11,ir13,ir15],1))
        ir23=self.conv23_ir_relu(ir23)
        ir25=self.conv25_ir(torch.cat([ir11,ir13,ir15],1))
        ir25=self.conv25_ir_relu(ir25)

        all_feature=torch.cat([vi21,vi23,vi25,ir21,ir23,ir25],1)

        d1=self.convall_1(all_feature)
        d1=self.convall_1_relu(d1)

        d2=self.convall_2(d1)
        d2=self.convall_2_relu(d2)

        d3=self.convall_3(torch.cat([d1,d2],1))
        d3=self.convall_3_relu(d3)
   
        d4=self.convall_4(torch.cat([d1,d2,d3],1))
        d4=self.convall_4_relu(d4)
      
        return d4




class ConvFusion(nn.Module):
    def __init__(self):
        super(ConvFusion, self).__init__()
        self.tanh = nn.Tanh()
        self.sft1_vi=SFTLayer()
        self.sft2_vi=SFTLayer()
        self.sft1_ir=SFTLayer()
        self.sft2_ir=SFTLayer()
        self.sift=SFTLayer()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        
        self.upsample=nn.PixelShuffle(2)
        self.pool=torch.nn.AvgPool2d(2,2)

        self.conv_1_vi1 = ConvLayer(32, 32, 3, 1,is_last=False)
        
        self.conv_1_vi2 = ConvLayer(32, 32, 3, 1,is_last=False)
        
        self.conv_1_deconv = ConvLayer(32, 32*4, 1, 1,is_last=False)
        
        self.conv_1_ir1 = ConvLayer(32, 32, 3, 1,is_last=False)

        self.conv_1_ir2 = ConvLayer(32, 32, 3, 1,is_last=False)
        
        self.conv_1_all_1 = ConvLayer(64, 64, 3, 1,is_last=False)
        
        self.conv_1_all_2 = ConvLayer(64, 32, 3, 1,is_last=False)
        self.conv_1_all_3 = ConvLayer(32, 32, 3, 1,is_last=False)
        self.conv_1_all_4 = ConvLayer(32, 32, 3, 1,is_last=False)
        
        self.conv_2_vi1 = ConvLayer(32, 64, 3, 1)
        
        self.conv_2_vi2 = ConvLayer(64, 32, 3, 1)
        
        self.conv_2_ir1 = ConvLayer(32, 64, 3, 1)

        self.conv_2_ir2 = ConvLayer(64, 32, 3, 1)
        
        self.conv_2_all_1 = ConvLayer(64, 32, 3, 1)
        
        self.conv_2_all_2 = ConvLayer(32, 32, 3, 1)
        
        
        self.conv_3_vi1 = ConvLayer(32, 64, 3, 1)
        
        self.conv_3_vi2 = ConvLayer(64, 32, 3, 1)
        
        self.conv_3_ir1 = ConvLayer(32, 64, 3, 1)

        self.conv_3_ir2 = ConvLayer(64, 32, 3, 1)
        
        self.conv_3_all_1 = ConvLayer(64, 32, 3, 1)
        
        self.conv_3_all_2 = ConvLayer(32, 32, 3, 1)
        
        
        self.conv_4_vi1 = ConvLayer(32, 64, 3, 1)
        
        self.conv_4_vi2 = ConvLayer(64, 32, 3, 1)
        
        self.conv_4_ir1 = ConvLayer(32, 64, 3, 1)

        self.conv_4_ir2 = ConvLayer(64, 32, 3, 1)
        
        self.conv_4_all_1 = ConvLayer(64, 32, 3, 1)
        
        self.conv_4_all_2 = ConvLayer(32, 32, 3, 1)
               

    def forward(self,vi,ir):
    
        vi1=self.conv_1_vi1(vi)
        vi1=self.relu1(vi1)
        
        ir1=self.conv_1_ir1(ir)
        ir1=self.relu3(ir1)
        
        vi_sift1=self.sft1_vi(vi1,ir1)
        ir_sift1=self.sft1_ir(ir1,vi1)
        
        vi2=self.conv_1_vi2(vi_sift1)
        vi2=self.relu2(vi2)
        
        ir2=self.conv_1_ir2(ir_sift1)
        ir2=self.relu4(ir2)
        
        vi_sift2=self.sft2_vi(vi2,ir2)
        ir_sift2=self.sft2_ir(ir2,vi2)
        
        all1=torch.cat([vi_sift2,ir_sift2],1)
        all1=self.conv_1_all_1(all1)
        all1=self.relu5(all1)
        all1=self.conv_1_all_2(all1)
        all1=self.relu6(all1)     
        
        return all1
                
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32, ratio=16):
        super(ChannelAttention, self).__init__()
        #平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        #MLP  除以16是降维系数
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) #kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #结果相加
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        #声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  #平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True) #最大池化
        #拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x) #7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)

class network_vi(nn.Module):
    def __init__(self):
        super(network_vi, self).__init__()
        self.vi_encoder=network_vi_encoder()
        self.vi_decoder=network_vi_decoder()
        self.sig = nn.Sigmoid()
    def forward(self,vi):
        vi1,vi2,vi3,vi4=self.vi_encoder(vi)	
        vi_result=self.vi_decoder(vi1,vi2,vi3,vi4)
        return vi_result
        
class network_ir(nn.Module):
    def __init__(self):
        super(network_ir, self).__init__()
        self.ir_encoder=network_ir_encoder()
        self.ir_decoder=network_ir_decoder()
        self.sig = nn.Sigmoid()
    def forward(self,ir):
        ir1,ir2,ir3,ir4=self.ir_encoder(ir)	
        ir_result=self.ir_decoder(ir1,ir2,ir3,ir4)
        return ir_result
        
class network_fusion(nn.Module):
    def __init__(self):
        super(network_fusion, self).__init__()
        self.feature_fusion =feature_fusion()
        self.vr=network_vr()
    def forward(self,vi1_dct,ir1_dct,vi2_dct,ir2_dct,vi3_dct,ir3_dct,vi4_dct,ir4_dct):
        f1_1,f2_1,f3_1,f4_1=self.feature_fusion(vi1_dct,ir1_dct,vi2_dct,ir2_dct,vi3_dct,ir3_dct,vi4_dct,ir4_dct)
        return f1_1,f2_1,f3_1,f4_1    
         




       
class network_decoder_d(nn.Module):
    def __init__(self):
        super(network_decoder_d, self).__init__()
        #self.channel_attention_b = ChannelAttention()
        #self.spatial_attention_b = SpatialAttention()
        self.tanh = nn.Tanh()
        #self.conv_output_b = ConvLayer(64, 32, 3, 1, is_last=False)
        #self.conv_d6 = ConvLayer(96, 96, 3, 1)
        
        #self.conv_d5 = ConvLayer(128, 96, 3, 1)
       
        self.conv_d4 = ConvLayer(64, 64, 3, 1)
        
        self.conv_d3 = ConvLayer(64, 32, 3, 1)
        
        self.conv_d2 = ConvLayer(32, 16, 3, 1)

        self.conv_d1 = ConvLayer(16, 1, 3, 1,is_last=False)


    def forward(self,ir1,ir2):
        #b_feature=torch.max(shallow1,shallow2)
        #channel_feature_b = b_feature*self.channel_attention_b(b_feature)
        #spatial_feature_b = b_feature*self.spatial_attention_b(b_feature)

        #fusion_output_b=torch.cat([channel_feature_b,spatial_feature_b],1)
        #fusion_output_b=self.conv_output_b(fusion_output_b)

        #fusion_output_b=self.tanh_b(fusion_output_b)
        ir_all=torch.cat([ir1,ir2],1)
       
        #ir_d5=self.conv_d5(ir_all)

        ir_d4=self.conv_d4(ir_all)
        
        #ir_d54=torch.cat([ir_d5,ir_d4],1)

        ir_d3=self.conv_d3(ir_d4)
        
        #ir_d543=torch.cat([ir_d5,ir_d4,ir_d3],1)

        ir_d2=self.conv_d2(ir_d3)
        
        result_image=self.conv_d1(ir_d2)
        #result_image=self.tanh(result_image)
        #result_image=result_image/2+0.5
        
        return result_image
        
class network_decoder_s(nn.Module):
    def __init__(self):
        super(network_decoder_s, self).__init__()
        #self.channel_attention_b = ChannelAttention()
        #self.spatial_attention_b = SpatialAttention()
        self.tanh = nn.Tanh()
        #self.conv_output_b = ConvLayer(64, 32, 3, 1, is_last=False)
        #self.conv_d6 = ConvLayer(96, 96, 3, 1)
        
        #self.conv_d5 = ConvLayer(128, 96, 3, 1)
       
        self.conv_d4 = ConvLayer(64, 64, 3, 1)
        
        self.conv_d3 = ConvLayer(64, 32, 3, 1)
        
        self.conv_d2 = ConvLayer(32, 16, 3, 1)

        self.conv_d1 = ConvLayer(16, 1, 3, 1,is_last=False)


    def forward(self,ir1,ir2):
        #b_feature=torch.max(shallow1,shallow2)
        #channel_feature_b = b_feature*self.channel_attention_b(b_feature)
        #spatial_feature_b = b_feature*self.spatial_attention_b(b_feature)

        #fusion_output_b=torch.cat([channel_feature_b,spatial_feature_b],1)
        #fusion_output_b=self.conv_output_b(fusion_output_b)

        #fusion_output_b=self.tanh_b(fusion_output_b)
        ir_all=torch.cat([ir1,ir2],1)
       
        #ir_d5=self.conv_d5(ir_all)

        ir_d4=self.conv_d4(ir_all)
        
        #ir_d54=torch.cat([ir_d5,ir_d4],1)

        ir_d3=self.conv_d3(ir_d4)
        
        #ir_d543=torch.cat([ir_d5,ir_d4,ir_d3],1)

        ir_d2=self.conv_d2(ir_d3)
        
        result_image=self.conv_d1(ir_d2)
        #result_image=self.tanh(result_image)
        #result_image=result_image/2+0.5
        
        return result_image       
        
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
class DAC(nn.Module):
    def __init__(self, n_channels):
        super(DAC, self).__init__()

        self.mean = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )
        self.std = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )

    def forward(self, observed_feat, referred_feat):
        assert (observed_feat.size()[:2] == referred_feat.size()[:2])
        size = observed_feat.size()
        #print(size)
        referred_mean, referred_std = calc_mean_std(referred_feat)
        observed_mean, observed_std = calc_mean_std(observed_feat)
        #print(observed_mean.size())
        normalized_feat = (observed_feat - observed_mean.expand(
            size)) / observed_std.expand(size)
        referred_mean = self.mean(referred_mean)
        referred_std = self.std(referred_std)
        output = normalized_feat * referred_std.expand(size) + referred_mean.expand(size)
        return output


class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        #print(hessian.size())
        #for p in range(0,32):
            #file_name = str(p)+'.png'
            #output_path1 = '/home/liu/ICIP/method_version1/feature_vis/' + file_name
            #utils.save_image_test(fxx[:,p,:,:], output_path1)
        return hessian


class rcab_block(nn.Module):
    def __init__(self, n_channels, kernel, bias=False, activation=nn.ReLU(inplace=True)):
        super(rcab_block, self).__init__()

        block = []

        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))
        block.append(activation)
        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))

        self.block = nn.Sequential(*block)

        self.calayer = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        residue = self.block(x)
        chnlatt = F.adaptive_avg_pool2d(residue, 1)
        chnlatt = self.calayer(chnlatt)
        output = x + residue * chnlatt

        return output


class DiEnDec(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super(DiEnDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.gate(self.decoder(self.encoder(x)))
        return output


class SingleModule(nn.Module):
    def __init__(self, n_channels, n_blocks, act, attention):
        super(SingleModule, self).__init__()
        res_blocks = [rcab_block(n_channels=n_channels, kernel=3, activation=act) for _ in range(n_blocks)]
        self.body_block = nn.Sequential(*res_blocks)
        self.attention = attention
        if attention:
            self.coder = nn.Sequential(DiEnDec(3, act))
            self.dac = nn.Sequential(DAC(n_channels))
            self.hessian3 = nn.Sequential(MSHF(n_channels, kernel=3))
            self.hessian5 = nn.Sequential(MSHF(n_channels, kernel=5))
            self.hessian7 = nn.Sequential(MSHF(n_channels, kernel=7))

    def forward(self, x):
        sz = x.size()
        resin = self.body_block(x)

        if self.attention:
            hessian3 = self.hessian3(resin)
            hessian5 = self.hessian5(resin)
            hessian7 = self.hessian7(resin)
            hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                                 torch.mean(hessian5, dim=1, keepdim=True),
                                 torch.mean(hessian7, dim=1, keepdim=True))
                                , 1)
            hessian = self.coder(hessian)
            attention = torch.sigmoid(self.dac[0](hessian.expand(sz), x))
            resout = resin * attention
        else:
            resout = resin

        output = resout #+ x

        return output






