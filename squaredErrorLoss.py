# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:04:51 2023

@author: user
"""
import torch
#import torch.nn as nn
#import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb


class SquaredErrorLoss(torch.nn.Module):
    def __init__(self):
        super(SquaredErrorLoss, self).__init__()
        
    def forward(self, y_fcn, im_data, im_info, gt_boxes, num_boxes): # X, Y, Classes
    
        input_shape = torch.tensor(im_data.size())
        n = im_data.shape[1]
        index = torch.tensor((range(0,n))).to(im_data.device)
        
        Error_sum = torch.tensor(0.0)
        
        for batch in range(len(num_boxes)):
            for gt_box in gt_boxes[batch, 0:num_boxes[batch], :].type(torch.int16):
                box, clg = gt_box[0:4], gt_box[4]
                
                # gt_box [x1,y1,x2,y2,clg]
                # print(batch, box, clg)
                Y_class = y_fcn[batch, index+n*clg, :,:] 
                
                # plt.imshow(1+(im_data[batch,:,box[1]:box[3], box[0]:box[2]])
                #            .permute([1,2,0]).cpu().detach().numpy()/100)
                
                Error_value = torch.pow(im_data[batch,:,box[1]:box[3], box[0]:box[2]] - 
                                  Y_class[:,box[1]:box[3], box[0]:box[2]], 2).sum() \
                                  / (n * (box[3] - box[1]) * (box[2] - box[0]))
                
                Error_sum = Error_sum + Error_value
        
        return Error_sum / num_boxes.sum()
    
# FCN 损失和优化函数
loss_fcn = SquaredErrorLoss()
#loss_fcn(error_map, im_data, im_info, gt_boxes, num_boxes)


