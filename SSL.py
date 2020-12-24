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


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

CUDA_DIVICE_ID = '0'
RATIO_LABEL = 0.5

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default='VGG',
                        help="available options : ResNet, VGG.")
    parser.add_argument("--data-dir", type=str, default='/data1/CityScapes',
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default='./dataset/cityscapes_list/train.txt',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default='./snapshots/BestGTA5.pth',                ##change the directory for your data
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default='train',
                        help="choose evaluation set.")
    parser.add_argument("--ratio_label", type=float, default=RATIO_LABEL,
                        help="the ratio of pseudo labels.")
    parser.add_argument("--save", type=str, default='./pseudo_label/',
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)
    return parser.parse_args()


def main():
    #args = opt.initialize()  
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'ResNet':
        model = DeeplabMulti(num_classes=args.num_classes)
    if args.model == 'VGG':
        model = DeeplabVGG(num_classes=args.num_classes)

    device = torch.device("cuda" if not args.cpu else "cpu")

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.to(device)  
    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))
    predicted_prob_for_cbst = np.zeros((len(targetloader), 512, 1024,19), dtype=np.float16)
    image_name = []
    
    for index, batch in enumerate(targetloader):

        if index % 100 == 0:
            print( '%d processd' % index)
        image, _, name = batch
        image = image.to(device)

        with torch.no_grad():
            _, output = model(image)
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1,2,0)

        predicted_prob_for_cbst[index] = output.copy()
        
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])
        
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(1)                               
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*(1-args.ratio_label)))])          
    print (thres)
    thres = np.array(thres)
    thres = thres[np.newaxis, np.newaxis, :]

    for index in range(len(targetloader)):
        name = image_name[index]
        output = predicted_prob_for_cbst[index]/thres                       #normalization the prob 
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        for i in range(19):
            label[(prob<1)*(label==i)] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (args.save, name)) 
    
    
if __name__ == '__main__':
    main()
    