import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
from torch.utils import data
from dataset.cityscapes_dataset import cityscapesDataSet
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from model.deeplab_multi_reduce import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset_label import cityscapesDataSetLabel
import torch.nn.functional as F
from model.function import class_center_precal
import torchsnooper


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
CUDA_DIVICE_ID = '1'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default='ResNet',
                        help="available options : ResNet, VGG.")
    parser.add_argument("--data-dir", type=str, default='/data1/CityScapes',
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default='./dataset/cityscapes_list/train.txt',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default='./snapshots/BestGTA5.pth',        #change the directory for your data
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default='train',
                        help="choose evaluation set.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)
    return parser.parse_args()


   

def main():

    #args = opt.initialize()  
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

    if args.model == 'ResNet':
        model = DeeplabMulti(num_classes=args.num_classes)
    if args.model == 'VGG':
        model = DeeplabVGG(num_classes=args.num_classes)

    device = torch.device("cuda" if not args.cpu else "cpu")

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.to(device)  
    
    targetloader_center = data.DataLoader(cityscapesDataSetLabel(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    count_class = np.zeros((19, 1))
    class_center_temp = np.zeros((19, 256))
    
    
    for index, batch in enumerate(targetloader_center):
        if index % 100 == 0:
            print( '%d processd' % index)
        images, labels, _, _= batch
        images = images.to(device)
        labels = labels.long().to(device)
        

        with torch.no_grad():
            feature, _ = model(images)
        
        class_center,count_class_t = class_center_precal(feature,labels)
        count_class += count_class_t.numpy()
        class_center_temp += class_center.cpu().data[0].numpy()

    
    
    count_class[count_class==0] = 1              #in case divide 0 error
    
    class_center = class_center_temp/count_class
    np.save('./target_center.npy',class_center)
    
    
    
if __name__ == '__main__':

    main()
    