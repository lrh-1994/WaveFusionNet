import os
import numpy as np
from util.Evaluator import Evaluator
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from util.img_read_save import img_save,image_read_cv2
import time
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for dataset_name in ["MSRS"]:

    print("\n"*2+"="*80)

    print("The test result of "+dataset_name+' :')

    test_folder=os.path.join('test_data/',dataset_name) 

    test_out_folder=os.path.join('results/',dataset_name) 

    eval_folder=test_out_folder
      
    ori_img_folder=test_folder

    metric_results = np.zeros((6))

    for img_name in os.listdir(os.path.join(ori_img_folder,"IR/")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"IR/", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"VI/", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_results += np.array([Evaluator.EN(fi), 
                                        Evaluator.SD(fi),
                                        Evaluator.SF(fi), 
                                        Evaluator.SCD(fi,ir, vi),
                                        Evaluator.VIFF(fi, ir, vi),
                                        Evaluator.Qabf(fi,ir, vi)])
            
    metric_results /= len(os.listdir(eval_folder))

    print("\t\t EN\t SD\t SF\t SCD\t VIF\t Qabf")
    print('results'+'\t\t'+str(np.round(metric_results[0], 2))+'\t'
            +str(np.round(metric_results[1], 2))+'\t'
            +str(np.round(metric_results[2], 2))+'\t'
            +str(np.round(metric_results[3], 2))+'\t'
            +str(np.round(metric_results[4], 2))+'\t'
            +str(np.round(metric_results[5], 2))
            )
    print("="*80)

for dataset_name in ["RoadScene"]:

    print("\n"*2+"="*80)

    print("The test result of "+dataset_name+' :')

    test_folder=os.path.join('test_data/',dataset_name) 

    test_out_folder=os.path.join('results/',dataset_name) 

    eval_folder=test_out_folder
      
    ori_img_folder=test_folder

    metric_results = np.zeros((6))

    for img_name in os.listdir(os.path.join(ori_img_folder,"IR/")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"IR/", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"VI/", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_results += np.array([Evaluator.EN(fi), 
                                        Evaluator.SD(fi),
                                        Evaluator.SF(fi), 
                                        Evaluator.SCD(fi,ir, vi),
                                        Evaluator.VIFF(fi, ir, vi),
                                        Evaluator.Qabf(fi,ir, vi)])
            
    metric_results /= len(os.listdir(eval_folder))

    print("\t\t EN\t SD\t SF\t SCD\t VIF\t Qabf")
    print('results'+'\t\t'+str(np.round(metric_results[0], 2))+'\t'
            +str(np.round(metric_results[1], 2))+'\t'
            +str(np.round(metric_results[2], 2))+'\t'
            +str(np.round(metric_results[3], 2))+'\t'
            +str(np.round(metric_results[4], 2))+'\t'
            +str(np.round(metric_results[5], 2))
            )
    print("="*80)

for dataset_name in ["TNO"]:

    print("\n"*2+"="*80)

    print("The test result of "+dataset_name+' :')

    test_folder=os.path.join('test_data/',dataset_name) 

    test_out_folder=os.path.join('results/',dataset_name) 

    eval_folder=test_out_folder
      
    ori_img_folder=test_folder

    metric_results = np.zeros((6))

    for img_name in os.listdir(os.path.join(ori_img_folder,"IR/")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"IR/", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"VI/", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_results += np.array([Evaluator.EN(fi), 
                                        Evaluator.SD(fi),
                                        Evaluator.SF(fi), 
                                        Evaluator.SCD(fi,ir, vi),
                                        Evaluator.VIFF(fi, ir, vi),
                                        Evaluator.Qabf(fi,ir, vi)])
            
    metric_results /= len(os.listdir(eval_folder))

    print("\t\t EN\t SD\t SF\t SCD\t VIF\t Qabf")
    print('results'+'\t\t'+str(np.round(metric_results[0], 2))+'\t'
            +str(np.round(metric_results[1], 2))+'\t'
            +str(np.round(metric_results[2], 2))+'\t'
            +str(np.round(metric_results[3], 2))+'\t'
            +str(np.round(metric_results[4], 2))+'\t'
            +str(np.round(metric_results[5], 2))
            )
    print("="*80)



