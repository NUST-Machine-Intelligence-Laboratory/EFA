import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchsnooper
import time

#device = torch.device("cuda" if not args.cpu else "cpu")
device = torch.device("cuda")

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())



def class_center_cal(feature, labels,num_classes=19):   
    labels = labels.to(device)
    labels = torch.unsqueeze(labels,1) 
    labels = F.interpolate(labels,scale_factor=0.5)
    labels = torch.squeeze(labels,1)
    labels = labels.long()
    n_l,h_l,w_l = labels.size()
    labels = labels.view(n_l,h_l*w_l)
    labels_for_count = labels.cpu().data[0].numpy()
    labels_for_count[labels_for_count==255] = 19  #255->19
    count_class= np.bincount(labels_for_count, minlength=19)
    count_class = count_class[:19]
    count_class = torch.from_numpy(count_class).unsqueeze(1)                                  
    labels[labels==255] = 19  #255->19   
    labels = torch.unsqueeze(labels,2).cpu().long()
    labels = torch.zeros(n_l,h_l*w_l,num_classes+1).scatter_(2,labels,1)  #n  h*w  20
    labels = labels.transpose(1,2) #n 20 h*w
    labels = labels[:,:num_classes,:]  #n 19 h*w
    feature = nn.functional.interpolate(feature,size=(h_l, w_l), mode='bilinear', align_corners=True)   
    n,c,h,w = feature.size()
    feature = F.softmax(feature,1)                    #softmax
    feature = feature.view(n,c,-1)    
    feature = feature.transpose(1,2).cpu()  #n h*w c   
    class_center = torch.matmul(labels,feature)  #n 19 c   
    class_center = class_center[0]
    count_class[count_class==0] = 1              #in case divide 0 error
    class_center = class_center/count_class.float()    
    return class_center           #19  c


def class_center_update(class_center_source, class_center_source_ori,p,num_classes=19):
    for i in range(num_classes):
        x = class_center_source[i]
        x = x[x>0]
        if len(x)>0 :
            class_center_source[i] = p*class_center_source[i] + (1-p)*class_center_source_ori[i].float()
        else :  
            class_center_source[i] = class_center_source_ori[i] 

    return class_center_source



def class_center_precal(feature,labels,num_classes=19):
    count_class = np.zeros((19, 1))
    labels = labels.to(device)
    n,c,h,w = feature.size()
    labels = F.interpolate(labels.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).long()
    n_l,h_l,w_l = labels.size()
    labels = labels.view(n_l,h_l*w_l)
    labels_for_count = labels.cpu().data[0].numpy()
    labels_for_count[labels_for_count==255] = 19  #255->19
    count_class= np.bincount(labels_for_count, minlength=19)
    count_class = count_class[:19]
    count_class = torch.from_numpy(count_class).unsqueeze(1)                                      
    labels[labels==255] = 19  #255->19
    labels = torch.unsqueeze(labels,2).cpu()
    labels = torch.zeros(n,h_l*w_l,num_classes+1).scatter_(2,labels,1)
    labels = labels.transpose(1,2) #n 20 h*w
    labels = labels[:,:num_classes,:]  #n 19 h*w
    feature = nn.functional.interpolate(feature,size=(h_l, w_l), mode='bilinear', align_corners=True)
    feature = F.softmax(feature,1)                    #softmax
    feature = feature.view(n,c,-1)
    feature = feature.transpose(1,2).cpu()  #n h*w c
    class_center = torch.matmul(labels,feature)  #n 19 c

    return class_center,count_class






