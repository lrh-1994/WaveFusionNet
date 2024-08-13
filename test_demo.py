# test phase
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward,DWTInverse
#from transformer import trans_fuse
from torch.autograd import Variable
from fusion_strategy import average,maximum,sumall,l1_attention
from net_update import network_fusion,network_vi,network_ir
import utils
from args import args
import numpy as np
eps = 1e-10
import time

def run_demo(model_fusion , ir_path, vi_path, output_path, name_ir, flag_img):
	img_ir, h, w, c = utils.get_test_image(ir_path, args.HEIGHT, args.WIDTH, flag=flag_img)
	img_vi, h, w, c = utils.get_test_image(vi_path, args.HEIGHT, args.WIDTH, flag=flag_img)
	if args.cuda:
		model_fusion.cuda()
		img_ir = img_ir.cuda()
		img_vi = img_vi.cuda()
	
	dwt = DWTForward(J=1,mode='zero',wave='haar').cuda()
	idwt = DWTInverse(mode='zero',wave='haar').cuda()
	
	a=torch.mean(img_vi,[2,3],keepdim=True)
	b=torch.mean(img_ir,[2,3],keepdim=True)
	img_vi_b=img_vi-torch.mean(img_vi,[2,3],keepdim=True)
	img_ir_b=img_ir-torch.mean(img_ir,[2,3],keepdim=True)
	
	img_ratio=torch.zeros(1,1,img_vi.size(2),img_vi.size(3)).cuda()
	
	img_out = torch.zeros(1,1,img_vi.size(2),img_vi.size(3)).cuda()
	
	tensor1=torch.zeros(1,1,img_vi.size(2),img_vi.size(3)).cuda()+1
	tensor0=torch.zeros(1,1,img_vi.size(2),img_vi.size(3)).cuda()
	
	f64_h=torch.rand((1,32,3,64,64),requires_grad=False).cuda()
	f32_h=torch.rand((1,32,3,32,32),requires_grad=False).cuda()
	f16_h=torch.rand((1,32,3,16,16),requires_grad=False).cuda()
	
	model_fusion.eval()
	for i in range(0,(h-128),120):
		for j in range(0,(w-128),120):
			if c is 1:
				vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h = model_fusion.vr.vi_encoder(img_vi_b[:,:,i:i+128,j:j+128],dwt)
				ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h = model_fusion.vr.ir_encoder(img_ir_b[:,:,i:i+128,j:j+128],dwt)
			
				com_vi=torch.ge(torch.abs(vi64_h[0]),torch.abs(ir64_h[0])).float()
				com_ir=torch.ge(torch.abs(ir64_h[0]),torch.abs(vi64_h[0])).float()
				f64_h_1=com_vi*vi64_h[0]+com_ir*ir64_h[0]
			
				f64_h=[f64_h_1]
			
			
				com_vi=torch.ge(torch.abs(vi32_h[0]),torch.abs(ir32_h[0])).float()
				com_ir=torch.ge(torch.abs(ir32_h[0]),torch.abs(vi32_h[0])).float()
				f32_h_1=com_vi*vi32_h[0]+com_ir*ir32_h[0]
			
				f32_h=[f32_h_1]
			
			
				com_vi=torch.ge(torch.abs(vi16_h[0]),torch.abs(ir16_h[0])).float()
				com_ir=torch.ge(torch.abs(ir16_h[0]),torch.abs(vi16_h[0])).float()
				f16_h_1=com_vi*vi16_h[0]+com_ir*ir16_h[0]
			
				f16_h=[f16_h_1]
			
				f64_l,f32_l,f16_l = model_fusion.feature_fusion(vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l)	


				out=model_fusion.vr.decoder(f64_l,f32_l,f16_l,f64_h,f32_h,f16_h,idwt)
				
				img_out[:,:,i:i+128,j:j+128]=out+img_out[:,:,i:i+128,j:j+128]
				img_ratio[:,:,i:i+128,j:j+128]=img_ratio[:,:,i:i+128,j:j+128]+1
			else:
				a=1
	for i in range(0,(h-128),120):
		for j in range(w-128,w,128):
			if c is 1:
				vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h = model_fusion.vr.vi_encoder(img_vi_b[:,:,i:i+128,j:j+128],dwt)
				ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h = model_fusion.vr.ir_encoder(img_ir_b[:,:,i:i+128,j:j+128],dwt)
			
				com_vi=torch.ge(torch.abs(vi64_h[0]),torch.abs(ir64_h[0])).float()
				com_ir=torch.ge(torch.abs(ir64_h[0]),torch.abs(vi64_h[0])).float()
				f64_h_1=com_vi*vi64_h[0]+com_ir*ir64_h[0]
			
				f64_h=[f64_h_1]
			
			
				com_vi=torch.ge(torch.abs(vi32_h[0]),torch.abs(ir32_h[0])).float()
				com_ir=torch.ge(torch.abs(ir32_h[0]),torch.abs(vi32_h[0])).float()
				f32_h_1=com_vi*vi32_h[0]+com_ir*ir32_h[0]
			
				f32_h=[f32_h_1]
			
			
				com_vi=torch.ge(torch.abs(vi16_h[0]),torch.abs(ir16_h[0])).float()
				com_ir=torch.ge(torch.abs(ir16_h[0]),torch.abs(vi16_h[0])).float()
				f16_h_1=com_vi*vi16_h[0]+com_ir*ir16_h[0]
			
				f16_h=[f16_h_1]
			
				f64_l,f32_l,f16_l = model_fusion.feature_fusion(vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l)	

				
				out=model_fusion.vr.decoder(f64_l,f32_l,f16_l,f64_h,f32_h,f16_h,idwt)
				
				img_out[:,:,i:i+128,j:j+128]=out+img_out[:,:,i:i+128,j:j+128]
				img_ratio[:,:,i:i+128,j:j+128]=img_ratio[:,:,i:i+128,j:j+128]+1
				
			else:
				a=1

	for i in range(h-128,h,128):
		for j in range(0,w-128,120):
			if c is 1:
				vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h = model_fusion.vr.vi_encoder(img_vi_b[:,:,i:i+128,j:j+128],dwt)
				ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h = model_fusion.vr.ir_encoder(img_ir_b[:,:,i:i+128,j:j+128],dwt)
			
				com_vi=torch.ge(torch.abs(vi64_h[0]),torch.abs(ir64_h[0])).float()
				com_ir=torch.ge(torch.abs(ir64_h[0]),torch.abs(vi64_h[0])).float()
				f64_h_1=com_vi*vi64_h[0]+com_ir*ir64_h[0]
			
				f64_h=[f64_h_1]
			
			
				com_vi=torch.ge(torch.abs(vi32_h[0]),torch.abs(ir32_h[0])).float()
				com_ir=torch.ge(torch.abs(ir32_h[0]),torch.abs(vi32_h[0])).float()
				f32_h_1=com_vi*vi32_h[0]+com_ir*ir32_h[0]
			
				f32_h=[f32_h_1]
			
			
				com_vi=torch.ge(torch.abs(vi16_h[0]),torch.abs(ir16_h[0])).float()
				com_ir=torch.ge(torch.abs(ir16_h[0]),torch.abs(vi16_h[0])).float()
				f16_h_1=com_vi*vi16_h[0]+com_ir*ir16_h[0]
			
				f16_h=[f16_h_1]
			
				f64_l,f32_l,f16_l= model_fusion.feature_fusion(vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l)	
				
				
				out=model_fusion.vr.decoder(f64_l,f32_l,f16_l,f64_h,f32_h,f16_h,idwt)
				
				img_out[:,:,i:i+128,j:j+128]=out+img_out[:,:,i:i+128,j:j+128]
				img_ratio[:,:,i:i+128,j:j+128]=img_ratio[:,:,i:i+128,j:j+128]+1
			else:
				a=1
	
	vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h = model_fusion.vr.vi_encoder(img_vi_b[:,:,h-128:h,w-128:w],dwt)
	ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h = model_fusion.vr.ir_encoder(img_ir_b[:,:,h-128:h,w-128:w],dwt)
			
	com_vi=torch.ge(torch.abs(vi64_h[0]),torch.abs(ir64_h[0])).float()
	com_ir=torch.ge(torch.abs(ir64_h[0]),torch.abs(vi64_h[0])).float()
	f64_h_1=com_vi*vi64_h[0]+com_ir*ir64_h[0]
			
	f64_h=[f64_h_1]
			
			
	com_vi=torch.ge(torch.abs(vi32_h[0]),torch.abs(ir32_h[0])).float()
	com_ir=torch.ge(torch.abs(ir32_h[0]),torch.abs(vi32_h[0])).float()
	f32_h_1=com_vi*vi32_h[0]+com_ir*ir32_h[0]
			
	f32_h=[f32_h_1]
			
			
	com_vi=torch.ge(torch.abs(vi16_h[0]),torch.abs(ir16_h[0])).float()
	com_ir=torch.ge(torch.abs(ir16_h[0]),torch.abs(vi16_h[0])).float()
	f16_h_1=com_vi*vi16_h[0]+com_ir*ir16_h[0]
			
	f16_h=[f16_h_1]
	
	f64_l,f32_l,f16_l = model_fusion.feature_fusion(vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l)

				
	out=model_fusion.vr.decoder(f64_l,f32_l,f16_l,f64_h,f32_h,f16_h,idwt)
				
	
	img_out[:,:,h-128:h,w-128:w]=out+img_out[:,:,h-128:h,w-128:w]
	img_ratio[:,:,h-128:h,w-128:w]=img_ratio[:,:,h-128:h,w-128:w]+1

	ref_img=torch.max(img_vi,img_ir)
	img_out=torch.div(img_out,img_ratio)
	a=torch.mean(img_vi,[2,3],keepdim=True)
	b=torch.mean(img_ir,[2,3],keepdim=True)
	img_out = img_out+0.5*a+0.5*b
	
	img_out = torch.clamp(img_out, 0, 1)

	
	output_path_r = output_path + name_ir 
	utils.save_image_test(img_out[0,:,:,:], output_path_r)
	

def main():
	# False - gray True -RGB
	flag_img = False
	# ################# gray scale ########################################
	test_path = './test_data/TNO/IR/'#测试图像的路径
	
	model_stage2_path = './checkpoint.model'
	
	output_path = './results/TNO/' #融合结果的保存路径
	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	with torch.no_grad():
				arr=np.zeros((1,21),dtype=np.float64)
				model_fusion = network_fusion()
				model_fusion.load_state_dict(torch.load(model_stage2_path,map_location='cuda'))

				imgs_paths_ir, names = utils.list_images(test_path)
				num = len(imgs_paths_ir)
				for i in range(0,21):
					name_ir = names[i]
					ir_path = imgs_paths_ir[i]
					vi_path = ir_path.replace('IR', 'VI')
					tic = time.time()
					run_demo(model_fusion, ir_path, vi_path, output_path, name_ir, flag_img)
					toc = time.time()
					if i<21:
						arr[0,i]=toc-tic

				print(' visible and infrared Image Fusion Task Done......')
				
				print(arr[0,1:21])
				print(np.mean(arr[0,1:21]))
				print(np.std(arr[0,1:21]))

if __name__ == '__main__':
	main()




