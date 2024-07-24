import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes

#SL from model.roi_layers import ROIAlign, ROIPool
from torchvision.ops import RoIPool as ROIPool
from torchvision.ops import RoIAlign as ROIAlign
import  torchvision

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN_FCN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN_FCN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        self.model_fcn = models.segmentation.fcn_resnet101(pretrained=True,
                                                                    progress=True, 
                                                                    #num_classes= 21, 
                                                                    aux_loss=None)
        self.model_fcn.classifier[4] = torch.nn.Conv2d(512, 3 * cfg.num_classes, kernel_size=1,stride=1)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        #SL 加入fcn网络
        y_fcn = self.model_fcn(im_data)['out']
        error_map = torch.pow(torch.cat([im_data]* cfg.num_classes, axis=1) - y_fcn, 2)
        
        # pdb.set_trace()
        fcn_loss = self.SquaredErrorLoss(y_fcn, im_data, im_info, gt_boxes, num_boxes)/10000
            
        
        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
        else:
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            
            
        #FCN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0)
        #pool_fcn = FCN_roi_pool(error_map, rois.view(-1, 5)).sum([2,3])
        pool_fcn = torchvision.ops.roi_pool(input = error_map, 
                                            boxes = rois.view(-1, 5),
                                            output_size=[30,30],
                                            spatial_scale=1.0).sum([2,3])
        fcn_cls_score = pool_fcn[:,0::3] + pool_fcn[:,1::3] + pool_fcn[:,2::3]
        #fcn_cls_prob = F.softmax(fcn_cls_score, 1)
        fcn_cls_prob = F.softmin(fcn_cls_score, 1)
        #pdb.set_trace()
        

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        box_deltas = bbox_pred.data  #SL
        # if self.training and not self.class_agnostic:
        #     # select the corresponding columns according to roi labels
        #     bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        #     bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        #     bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat) 
        #cls_prob = F.softmax(cls_score, 1) #* fcn_cls_prob
        
        ###SL-----------------
        #scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
      
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            # box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              #if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(batch_size, -1, 4 * 16)  # len(imdb.classes)
      
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1,  cls_score.shape[1]))
      
        pred_boxes /= im_info[0][2].item()
        
        #scores = scores.view(batch_size,-1,16) # [batch, 300, 16]
        pred_boxes = pred_boxes.view(batch_size,-1, 64) # [batch, 300, 64]
        
        pool_fcn = []
        for batch in range(batch_size):
            pool_fcn_list = []
            for i in range(16):
                pool_fcn_list.append(\
                        torchvision.ops.roi_pool(input=error_map[batch:batch+1, i*3:(i+1)*3,:,:], 
                        boxes=[pred_boxes[batch,:,i*4:(i+1)*4]],
                        output_size=[50,50],
                        spatial_scale=1.0).sum([1,2,3]) )
            pool_fcn.append(torch.stack(pool_fcn_list,axis=1))
            
        fcn_score = torch.stack(pool_fcn, axis=0).view(-1, cls_score.shape[1]).cuda()
        fcn_cls_prob = F.softmin(fcn_score, 1)
        
        
        ###------------------
        cls_prob = F.softmax(cls_score * fcn_cls_prob, 1) 
        
        # compute bbox offset
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score* fcn_cls_prob, rois_label)  ### 

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
               RCNN_loss_cls, RCNN_loss_bbox, fcn_loss, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


    def SquaredErrorLoss(self, y_fcn, im_data, im_info, gt_boxes, num_boxes): # X, Y, Classes
    
        n = im_data.shape[1]
        index = torch.tensor((range(0,n))).to(im_data.device)
        
        Error_sum = torch.tensor(0.0)
        #pdb.set_trace()
        
        #box = [0,0,600,600]
        #Error_sum = (im_data.repeat(1,16,1,1) - y_fcn).pow(2).sum() \
        #              /im_data.shape[0]/ (n * (box[3] - box[1]) * (box[2] - box[0]))
        
        num = 0
        for batch in range(len(num_boxes)):
            if num_boxes[batch] == 0:
                continue
            for gt_box in gt_boxes[batch, 0:num_boxes[batch], :].type(torch.int64):
                box, clg = gt_box[0:4], gt_box[4]
                
                # gt_box [x1,y1,x2,y2,clg]
                # print(batch, box, clg, (n * (box[3] - box[1]) * (box[2] - box[0])))
                Y_class = y_fcn[batch, index+n*clg, :,:]
                
                # box = [0,0,600,600] #SL
                #pdb.set_trace()
                
                # plt.imshow(1+(im_data[batch,:,box[1]:box[3], box[0]:box[2]])
                #            .permute([1,2,0]).cpu().detach().numpy()/100)
                
                Error_value = torch.pow(im_data[batch,:,box[1]:box[3], box[0]:box[2]] - 
                                  Y_class[:,box[1]:box[3], box[0]:box[2]], 2).sum() \
                                  / (n * (box[3] - box[1]) * (box[2] - box[0]))
                #print(Error_value)
                Error_sum = Error_sum + Error_value
                num = num + 1
                
        Error_sum = Error_sum/ num_boxes.sum()
        
        torch.cuda.empty_cache()
        return Error_sum