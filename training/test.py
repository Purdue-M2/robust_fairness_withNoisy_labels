import os
import argparse
from detectors import DETECTOR
import torch
from torch.utils.data import DataLoader
from dataset.datasets_train import ImageDataset_Test_df
import csv
import time
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
from tqdm import tqdm
from log_utils import Logger
import os.path as osp
import sys
from fairness_metrics import acc_fairness
from transform import get_albumentations_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup_npy_files(directory):
    """
    Deletes all .npy files in the given directory.
    :param directory: The directory to clean up .npy files in.
    """
    for item in os.listdir(directory):
        if item.endswith(".npy"):
            os.remove(os.path.join(directory, item))
    print("Cleaned up .npy files in directory:", directory)

    # Function to adjust the keys of the state_dict for distributed-trained models
def adjust_state_dict(state_dict, ignore_keys=None):

    if ignore_keys is None:
        ignore_keys = []

    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix
        new_key = key.replace("module.", "") if key.startswith("module.") else key

        # Exclude keys in the ignore list
        if not any(ignored_key in new_key for ignored_key in ignore_keys):
            new_state_dict[new_key] = value

    return new_state_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batchsize", type=int,
                        default=32, help="size of the batches")
    parser.add_argument("--checkpoints", type=str,
                        default="./checkpoints/")
    parser.add_argument("--specific_checkpoint", type=str,
                        default="", help="specific checkpoint file to test")
    parser.add_argument("--inter_attribute", type=str,
                        default='male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others-young-middle-senior-ageothers')
    parser.add_argument("--savepath", type=str,
                        default='../results/')
    parser.add_argument("--model_structure", type=str, default='robust_pgfdd',
                        help="detector name")

    opt = parser.parse_args()
    sys.stdout = Logger(osp.join('../deepfake_results/dfdc/'+opt.model_structure+'/log_results.txt'))
    print(opt, '!!!!!!!!!!!')

    cuda = True if torch.cuda.is_available() else False
    test_transforms = get_albumentations_transforms([''])

    # Prepare the model (detector)
    model_class = DETECTOR[opt.model_structure]
    model = model_class()
    model.to(device)

    cleanup_npy_files(opt.savepath)

    # Build the path to the specific checkpoint
    specific_checkpoint_path = osp.join(opt.checkpoints, opt.specific_checkpoint)

    if not osp.exists(specific_checkpoint_path):
        print(f"Checkpoint not found: {specific_checkpoint_path}")
        exit(1)

    print(f"Testing specific checkpoint: {specific_checkpoint_path}")
    
    # Load and test the specific checkpoint
    checkpoint = torch.load(specific_checkpoint_path, weights_only=True)
    state_dict = adjust_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)

    interattributes = opt.inter_attribute.split('-')
    print("Model loaded and ready for testing.")

    
    csv_files = ['../dataset/dfdc/test_noise.csv']
    for csv_file in csv_files:
        print("%s" % csv_file)
        for eachatt in interattributes:
            test_dataset = ImageDataset_Test_df(csv_file, eachatt, test_transforms,'dfdc')

            test_dataloader = DataLoader(
                test_dataset, batch_size=opt.test_batchsize, shuffle=False, num_workers=8, pin_memory=True)
            
            print('Testing: ', eachatt, '%d batches int total' % len(test_dataloader))


            pred_list = []
            label_list = []

            for i, data_dict in enumerate(tqdm(test_dataloader)):

                model.eval()
                data, label = data_dict['image'], data_dict["label"]
                if 'label_spe' in data_dict:
                    data_dict.pop('label_spe')  
                data_dict['image'], data_dict["label"] = data.to(
                    device), label.to(device)

                with torch.no_grad():

                    output = model(data_dict, inference=True)
                    pred = output['cls']
                    pred = pred.cpu().data.numpy().tolist()

                    pred_list += pred
                    label_list += label.cpu().data.numpy().tolist()

            label_list = np.array(label_list)
            pred_list = np.array(pred_list)

            savepath = opt.savepath + '/' + eachatt
            np.save(savepath + 'labels.npy', label_list)
            np.save(savepath + 'predictions.npy', pred_list)

        acc_fairness(opt.savepath + '/', [['male', 'nonmale'], ['asian', 'white', 'black', 'others'],
                                        ['young', 'middle', 'senior', 'ageothers']])

        cleanup_npy_files(opt.savepath)

        print()
        print('-' * 10)
