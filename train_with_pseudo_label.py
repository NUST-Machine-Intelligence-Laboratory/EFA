import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from PIL import Image
from tensorboardX import SummaryWriter
from model.function import class_center_cal
from model.function import class_center_update
from model.deeplab_vgg import DeeplabVGG
from model.deeplab_multi_reduce import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.cityscapes_dataset_label import cityscapesDataSetLabel
from compute_iou import compute_mIoU
import csv

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/data0/gta5_deeplab'               #change the directory for your data
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = '/data1/CityScapes'          #change the directory for your data
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
DATA_LIST_PATH_TARGET_TEST = './dataset/cityscapes_list/val.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 120000
NUM_STEPS_STOP = 120000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './snapshots/BestGTA5.pth'       
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log2'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.0

LAMBDA_ADV_TARGET = 0.001
LAMBDA_S = 0.5
LAMBDA_CENTER = 0.05
LAMBDA_C_UPDATE = 0.01

TARGET = 'cityscapes'
SET = 'train'

MIOU_FILE = 'train'

CUDA_DIVICE_ID = '0'

SAVE_PATH = './result/'



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet,VGG")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-target-test", type=str, default=DATA_LIST_PATH_TARGET_TEST,
                        help="Path to the file listing the images in the target val dataset.")
    parser.add_argument("--miou-file", type=str, default=MIOU_FILE,
                        help="the name of csv file to save every mIoU.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target", type=float, default=LAMBDA_ADV_TARGET,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-s", type=float, default=LAMBDA_S,
                        help="lambda_s for auxiliary segmentation loss.")
    parser.add_argument("--lambda-center", type=float, default=LAMBDA_CENTER,
                        help="lambda_center for cross domain center alignment loss.")
    parser.add_argument("--lambda-center-update", type=float, default=LAMBDA_C_UPDATE,
                        help="lambda_center_update for center update.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)
    return parser.parse_args()


args = get_arguments()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    bestIoU = 0
    bestIter = 0 

    # Create network
    if args.model == 'ResNet':
        model = DeeplabMulti(num_classes=args.num_classes)
        saved_state_dict = torch.load(args.restore_from)
        model.load_state_dict(saved_state_dict)
    
    if args.model == 'VGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        saved_state_dict = torch.load(args.restore_from)
        model.load_state_dict(saved_state_dict)

    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    if args.model == 'ResNet':
        model_D = FCDiscriminator(num_classes=256).to(device)
        saved_state_dict = torch.load('./snapshots/BestGTA5_D.pth')     
        model_D.load_state_dict(saved_state_dict)
    if args.model == 'VGG':
        model_D = FCDiscriminator(num_classes=256).to(device)
        

    model_D.train()
    model_D.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)
    
    targetloader = data.DataLoader(cityscapesDataSetLabel(args.data_dir_target, args.data_list_target,
                                                          max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                          crop_size=input_size_target,
                                                          scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                          set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()    

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()


    bce_loss = torch.nn.BCEWithLogitsLoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    test_interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # load calculated  class center for initilization
    class_center_source_ori = np.load('./source_center.npy')         
    class_center_source_ori = torch.from_numpy(class_center_source_ori)

    class_center_target_ori = np.load('./target_center.npy')
    class_center_target_ori = torch.from_numpy(class_center_target_ori)

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):

        loss_seg = 0
        loss_adv_target_value = 0
        loss_D_value = 0
        loss_cla_value = 0
        loss_square_value = 0
        loss_st_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train G

        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # train with source

        _, batch = trainloader_iter.__next__()
        images, labels, _, _ = batch
        images = images.to(device)
        labels_s = labels  # copy for center calculation
        labels = labels.long().to(device)

        feature, prediction = model(images)  
        feature_s = feature  # copy for center calculation         
        prediction = interp(prediction)
        loss = seg_loss(prediction, labels)
        loss.backward(retain_graph=True)
        loss_seg = loss.item() 


        # train with target

        _, batch = targetloader_iter.__next__()
        images, labels_pseudo, _, _= batch
        labels_t = labels_pseudo  # copy for center calculation
        images = images.to(device)
        labels_pseudo = labels_pseudo.long().to(device)

        feature_target, pred_target = model(images)
        feature_t = feature_target  # copy for center calculation
        _,D_out = model_D(feature_target)
        loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        #print(args.lambda_adv_target)
        loss = args.lambda_adv_target * loss_adv_target
        loss.backward(retain_graph=True)
        loss_adv_target_value = loss_adv_target.item()

        pred_target = interp_target(pred_target)
        loss_st = seg_loss(pred_target, labels_pseudo)
        loss_st.backward(retain_graph=True)
        loss_st_value = loss_st.item()


        # class center alignment begin
        if i_iter > 10000 :
            class_center_source = class_center_cal(feature_s, labels_s)
            class_center_target = class_center_cal(feature_t, labels_t)
            class_center_source_ori = class_center_update(class_center_source, class_center_source_ori, args.lambda_center_update)
            class_center_target_ori = class_center_update(class_center_target, class_center_target_ori, args.lambda_center_update)

            class_center_source_ori = class_center_source_ori.detach()                                                #align target center to source

            center_diff = class_center_source_ori - class_center_target_ori
            loss_square = torch.pow(center_diff, 2).sum()

            loss = args.lambda_center * loss_square  
            loss.backward()
            loss_square_value = loss_square.item()
        # class center alignment end

        # train D

        # bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True

        # train with source
        feature = feature.detach()
        cla, D_out = model_D(feature)
        cla = interp(cla)
        loss_cla = seg_loss(cla, labels)
        
        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        loss_D = loss_D / 2
        #print(args.lambda_s)
        loss_Disc = args.lambda_s * loss_cla +loss_D
        loss_Disc.backward()

        loss_cla_value = loss_cla.item() 
        loss_D_value = loss_D.item()

        # train with target
        feature_target = feature_target.detach()           
        _,D_out = model_D(feature_target)
        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))            
        loss_D = loss_D / 2           
        loss_D.backward()          
        loss_D_value += loss_D.item()

        optimizer.step()        
        optimizer_D.step()

        class_center_target_ori = class_center_target_ori.detach()

        if args.tensorboard:
            scalar_info = {
                'loss_seg': loss_seg,
                'loss_cla': loss_cla_value,                
                'loss_adv_target': loss_adv_target_value,    
                'loss_st_value': loss_st_value,            
                'loss_D': loss_D_value,
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        #print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {3:.3f} loss_D = {4:.3f} loss_cla = {5:.3f} loss_st = {6:.5f} loss_square = {7:.5f}'.format(
            i_iter, args.num_steps, loss_seg, loss_adv_target_value, loss_D_value, loss_cla_value, loss_st_value, loss_square_value))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            if not os.path.exists(args.save):
                os.makedirs(args.save)
            testloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target_test, 
                                                           crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                         batch_size=1, shuffle=False, pin_memory=True)
            model.eval()
            for index, batch in enumerate(testloader):
                if index % 100 == 0:
                    print('%d processd' % index)
                image, _, name = batch
                with torch.no_grad():
                    output1, output2 = model(Variable(image).to(device))
                output = test_interp(output2).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                output = Image.fromarray(output)
                name = name[0].split('/')[-1]
                output.save('%s/%s' % (args.save, name))
            mIoUs = compute_mIoU(osp.join(args.data_dir_target,'gtFine/val'), args.save, 'dataset/cityscapes_list')
            mIoU = round(np.nanmean(mIoUs) * 100, 2)
            
            
            print('===>  current   mIoU: ' + str(mIoU))
            print('===> last best  mIoU: ' + str(bestIoU) )
            print('===> last best  iter: ' + str(bestIter))
            
            if mIoU > bestIoU:
                bestIoU = mIoU
                bestIter = i_iter
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'BestGTA5.pth'))
                torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'BestGTA5_D.pth'))
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))
            model.train()


    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
