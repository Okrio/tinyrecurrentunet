import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from dataset import load_CleanNoisyPairDataset
from stft_loss import MultiResolutionSTFTLoss
from util import rescale, find_max_epoch, print_size
from util import LinearWarmupCosineDecay, loss_fn

from network import TRUNet


def train(num_gpus, 
          rank, 
          group_name, 
          exp_path, 
          log, 
          optimization, 
          loss_config):

    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)
    
    # Create tensorboard logger.
    log_directory = os.path.join(log["directory"], exp_path)
    if rank == 0:
        tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # load training data
    trainloader = load_CleanNoisyPairDataset(**trainset_config, 
                            subset='training',
                            batch_size=optimization["batch_size_per_gpu"], 
                            num_gpus=num_gpus)
    print('Data loaded')
    
    # predefine model
    net = TRUNet(**network_config).cuda()
    print_size(net)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer AdamW
    optimizer = torch.optim.AdamW(net.parameters(), lr=optimization["learning_rate"])

    # load checkpoint
    time0 = time.time()
    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    else:
        ckpt_iter = log["ckpt_iter"]
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully')
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # training
    n_iter = ckpt_iter + 1

    # define learning rate scheduler and stft-loss
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=optimization["learning_rate"],
                    n_iter=optimization["n_iters"],
                    iteration=n_iter,
                    divider=25,
                    warmup_proportion=0.05,
                    phase=('linear', 'cosine'),
                )

    if loss_config["stft_lambda"] > 0:
        mrstftloss = MultiResolutionSTFTLoss(**loss_config["stft_config"]).cuda()
    else:
        mrstftloss = None

    while n_iter < optimization["n_iters"] + 1:
        
        # for each epoch
        for clean_feat, noisy_feat, fileid in trainloader: 
            
            #load data and send to device
            clean_feat = clean_feat.cuda()
            noisy_feat = noisy_feat.cuda()
            

            #forward propegation and loss calculation
            optimizer.zero_grad()
            X = (clean_feat, noisy_feat)
            
            loss, loss_dic = loss_fn(net, X, **loss_config, mrstftloss=mrstftloss)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            # back-propagation
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            scheduler.step()
            optimizer.step()

            # output to log
            if n_iter % log["iters_per_valid"] == 0:
                print("iteration: {} \treduced loss: {:.7f} \tloss: {:.7f}".format(
                    n_iter, reduced_loss, loss.item()), flush=True)
                
                if rank == 0:
                    # save to tensorboard
                    tb.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                    tb.add_scalar("Train/Train-Reduced-Loss", reduced_loss, n_iter)
                    tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                    tb.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter)
            
            # save checkpoint
            if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'iter': n_iter,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time()-time0)}, 
                            os.path.join(ckpt_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)
            print(n_iter)
            n_iter += 1
    # After training, close TensorBoard.
    if rank == 0:
        tb.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',     type=str, default='config.json', help='JSON file for configuration')
    parser.add_argument('-r', '--rank',       type=int, default=0,  help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    print(args.config)
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train"]       # training parameters
    
    global dist_config
    dist_config = config["dist"]         # to initialize distributed training
    
    global network_config
    network_config = config["network"]   # to define network
    print()
    global trainset_config
    trainset_config = config["trainset"] # to load trainset

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, 
          args.rank, 
          args.group_name, 
          **train_config)
