import sys
from detectors import DETECTOR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os.path as osp
from log_utils import Logger
import torch.backends.cudnn as cudnn
from dataset.pair_dataset import pairDatasetDro
from dataset.datasets_train import ImageDataset_Test
import csv
import argparse
from tqdm import tqdm
import time
import os
from utils.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM

from fairness_metrics import acc_fairness
from transform import get_albumentations_transforms

parser = argparse.ArgumentParser("Example")

parser.add_argument('--lr', type=float, default=0.0005,
                    help="learning rate for training")
parser.add_argument('--train_batchsize', type=int, default=16, help="batch size")
parser.add_argument('--test_batchsize', type=int, default=32, help="test batch size")
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--datapath', type=str,
                    default='../dataset/ff++/')
parser.add_argument("--continue_train", default=False, action='store_true')
parser.add_argument("--checkpoints", type=str, default='',
                    help="train model path")
parser.add_argument("--model", type=str, default='robust_pgfdd',
                    help="detector name[xception, pg_fdd,daw_fdd, efficientnetb4]")

parser.add_argument("--dataset_type", type=str, default='pair',
                    help="detector name[pair,no_pair]")

#################################test##############################

parser.add_argument("--inter_attribute", type=str,
                    default='male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others-young-middle-senior-ageothers')
parser.add_argument("--test_datapath", type=str,
                        default='../dataset/ff++/test_noise.csv', help="test data path")
parser.add_argument("--savepath", type=str,
                        default='../results')

args = parser.parse_args()


###### import data transform #######
from transform import fair_df_default_data_transforms as data_transforms
test_transforms = get_albumentations_transforms([''])

###### load data ######
if args.dataset_type == 'pair':
    train_dataset = pairDatasetDro(args.datapath + 'train_fake_spe.csv', args.datapath + 'train_real.csv', data_transforms['train'])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=16, pin_memory=True, collate_fn=train_dataset.collate_fn)
    train_dataset_size = len(train_dataset)



device = torch.device('cuda:5')

# prepare the model (detector)
model_class = DETECTOR[args.model]


def cleanup_npy_files(directory):
    """
    Deletes all .npy files in the given directory.
    :param directory: The directory to clean up .npy files in.
    """
    for item in os.listdir(directory):
        if item.endswith(".npy"):
            os.remove(os.path.join(directory, item))
    print("Cleaned up .npy files in directory:", directory)


# train and evaluation
def train(model,  optimizer_theta,optimizer_lambda, optimizer_p_list,scheduler, num_epochs, start_epoch):

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase = 'train'
        model.train()

        total_loss = 0.0

        for idx, data_dict in enumerate(tqdm(train_dataloader)):

            imgs, labels, intersec_labels = data_dict['image'], data_dict[
                'label'], data_dict['intersec_label']
            if 'label_spe' in data_dict:
                label_spe = data_dict['label_spe']
                data_dict['label_spe'] = label_spe.to(device)
            data_dict['image'], data_dict['label'], data_dict['intersec_label'] = imgs.to(
                device), labels.to(device), intersec_labels.to(device),

            with torch.set_grad_enabled(phase == 'train'):

                optimizer_lambda.zero_grad()
                for optimizer_p in optimizer_p_list:
                    optimizer_p.zero_grad()
                enable_running_stats(model)
                preds = model(data_dict)
                if isinstance(model, torch.nn.DataParallel):
                    losses,constraints = model.module.get_losses(data_dict, preds)
                else:
                    losses,constraints = model.get_losses(data_dict, preds)
                losses = losses['overall']
                lagrangian_loss = torch.dot(model.lambdas, constraints)
                losses = losses + lagrangian_loss
                
                losses.backward(retain_graph=True)
                optimizer_theta.first_step(zero_grad=True)


                disable_running_stats(model)
                preds = model(data_dict)
                if isinstance(model, torch.nn.DataParallel):
                    losses,constraints = model.module.get_losses(data_dict, preds)
                else:
                    losses,constraints = model.get_losses(data_dict, preds)
                losses = losses['overall']
                lagrangian_loss = torch.dot(model.lambdas, constraints)
                losses = losses + lagrangian_loss

                
                losses.backward(retain_graph=True)
                optimizer_theta.second_step(zero_grad=True)


                if model.lambdas.grad is not None:
                    # Perform gradient ascent by negating the gradient
                    model.lambdas.grad = -model.lambdas.grad

                optimizer_lambda.step()


                with torch.no_grad():
                    model.lambdas.data = model.project_lambdas(model.lambdas.data)
                for p in model.p_tildes.parameters():
                    if p.grad is not None:
                        p.grad = -p.grad
                for i, optimizer_p in enumerate(optimizer_p_list):
                    optimizer_p.step()
                    # Project p_tildes back to feasible space
                    with torch.no_grad():
                        current_batch_size = data_dict['phats'].size(0)
                        model.p_tildes[i].data[:current_batch_size] = model.project_ptilde(
                            data_dict, model.p_tildes[i].data[:current_batch_size], i
                        )

                

            if idx % 200 == 0:
                # compute training metric for each batch data
                if isinstance(model, torch.nn.DataParallel):
                    batch_metrics = model.module.get_train_metrics(data_dict, preds)
                else:
                    batch_metrics = model.get_train_metrics(data_dict, preds)


                print('#{} batch_metric{}'.format(idx, batch_metrics))

            total_loss += losses.item() * imgs.size(0)

        epoch_loss = total_loss / train_dataset_size
        print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))

        # update learning rate
        if phase == 'train':
            scheduler.step()

        # evaluation

        if (epoch+1) % 1 == 0:

            savepath = './checkpoints/'+args.model


            temp_model = savepath+"/"+args.model+str(epoch)+'.pth'
            torch.save(model.state_dict(), temp_model)

            print()
            print('-' * 10)

            phase = 'test'
            model.eval()

            interattributes = args.inter_attribute.split('-')


            for eachatt in interattributes:
                test_dataset = ImageDataset_Test(args.test_datapath, eachatt, test_transforms)

                test_dataloader = DataLoader(
                    test_dataset, batch_size=args.test_batchsize, shuffle=False,num_workers=32, pin_memory=True)

                print('Testing: ', eachatt)
                print('-' * 10)
                # print('%d batches int total' % len(test_dataloader))

                pred_list = []
                label_list = []

                for idx, data_dict in enumerate(tqdm(test_dataloader)):
                    imgs, labels = data_dict['image'], data_dict['label']
                    if 'label_spe' in data_dict:
                        data_dict.pop('label_spe')  # remove the specific label

                    data_dict['image'], data_dict['label'] = imgs.to(
                        device), labels.to(device)
                        
                    with torch.no_grad():
                        output = model(data_dict, inference=True)
                        pred = output['cls']
                        pred = pred.cpu().data.numpy().tolist()
            

                        pred_list += pred
                        label_list += labels.cpu().data.numpy().tolist()



                label_list = np.array(label_list)
                pred_list = np.array(pred_list)
                savepath = args.savepath + '/' + eachatt
                np.save(savepath+'labels.npy', label_list)
                np.save(savepath+'predictions.npy', pred_list)

                print()
                # print('-' * 10)
            acc_fairness(args.savepath + '/', [['male', 'nonmale'], ['asian', 'white', 'black', 'others'],['young', 'middle','senior','ageothers']])
            cleanup_npy_files(args.savepath)

    return model, epoch


def main():

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()


    sys.stdout = Logger(osp.join('./checkpoints/'+args.model+'/log_training.txt'))


    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    model = model_class()
    model.to(device)

    start_epoch = 0
    if args.continue_train and args.checkpoints != '':
        state_dict = torch.load(args.checkpoints)
        model.load_state_dict(state_dict)
        start_epoch = 15
        print(start_epoch)

    # optimize
    params_to_update = model.parameters()

    base_optimizer = torch.optim.SGD
    optimizer_theta  = SAM(params_to_update, base_optimizer,
                    lr=args.lr, momentum=0.9, weight_decay=5e-03)



    print(params_to_update, optimizer_theta)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_theta, step_size=60, gamma=0.9)

    num_groups = 8
    learning_rate_lambda = 0.0005
    learning_rate_p_list = [0.0005] * num_groups 

    # Ensure p_tildes are initialized
    dummy_data_dict = next(iter(train_dataloader))
    dummy_data_dict = {key: value.to(device) for key, value in dummy_data_dict.items()}  # Move data to the appropriate device
    model(dummy_data_dict)  # This will initialize p_tildes if not already initialized
    
    # Initialize optimizer for lambdas
    optimizer_lambda = optim.SGD([model.lambdas], lr=learning_rate_lambda, momentum=0.9, weight_decay=5e-3)

    # Initialize optimizers for p_tildes
    optimizer_p_list = [optim.SGD([p], lr=lr, momentum=0.9, weight_decay=5e-3) for p, lr in zip(model.p_tildes, learning_rate_p_list)]



    model, epoch = train(model, optimizer_theta,optimizer_lambda, optimizer_p_list,
                         exp_lr_scheduler, num_epochs=100, start_epoch=start_epoch)

    if epoch == 99:
        print("training finished!")
        exit()


if __name__ == '__main__':
    main()
