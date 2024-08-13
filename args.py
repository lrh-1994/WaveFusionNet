
class args():

	# training args
	epochs = 15 #"number of training epochs, default is 2"
	batch_size =8 #"batch size for training, default is 4"

	#dataset_vis = '/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/lrh/train_vis_128_2/'
	#dataset_ir = '/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/lrh/train_ir_128_2/'

	dataset_vis = '/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/infrared_visible/train_msrs_patch/msrs_vis_patch/'
	dataset_ir = '/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/infrared_visible/train_msrs_patch/msrs_ir_patch/'
	
	#dataset_vis = '/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/infrared_visible/msrs_256/msrs_256_vi/'
	#dataset_ir = '/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/infrared_visible/msrs_256/msrs_256_ir/'	

	HEIGHT = 128
	WIDTH = 128

	save_model = '/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/infrared_visible/fuxian/model/model_stage2_2.0_1/'
	save_loss = '/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/infrared_visible/fuxian/model/model_stage2_2.0_1/'

	image_size = 128 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	resume_network_vr='/media/mprl/06509c1a-c4d9-4e3e-9285-6da7c8d359b0/home/mprl/infrared_visible/model/stage1/Epoch_9_iters_600.model'
	lr = 4*1e-4 #2*1e-4 #"learning rate, default is 0.001"
	log_interval = 200  



