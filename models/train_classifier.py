import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

import utils.distributed as du
import utils.checkpoint as cu
from datasets.loader import shuffle_dataset
from models.criterion import t_criterion, max_criterion
from models.losses import l1_regularization, l2_regularization
from utils.eval_metrics import test_evaluate
from utils.config_defaults import labelLists
from calflops import calculate_flops
import psutil


@torch.no_grad()
def test(cfg, device, test_loader, model, criterion=nn.BCEWithLogitsLoss(reduction="none"), log_prefix=None,):
    start_time = time.time()

    model.eval()

    avg_loss = 0.

    all_idxs = []
    all_labels = []
    all_preds = []

    for cur_iter, (idx, inputs, labels) in enumerate(test_loader):
        iter_start_time = time.time()

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(inputs)

        loss = criterion(preds, labels).mean(0)
        avg_loss += loss

        preds = preds.cpu()
        labels = labels.cpu()

        m = nn.Sigmoid()
        preds = m(preds)

        all_idxs.extend(idx.tolist())
        all_labels.append(labels)
        all_preds.append(torch.unsqueeze(preds, 0) if preds.dim() == 1 else preds)



        if du.get_world_size() > 1:
            [loss] = du.all_reduce([loss])

        if du.is_master_proc():

            if (cur_iter + 1) % cfg.CLASSIFIER.PRINT_STEPS == 0:
                print("Test Model, {}/{}, Losses: {}, Avg Loss: {}, Time consuming: {:.2f}, Total time consuming: {:.2f}".format(
                     cur_iter, len(test_loader), loss, loss.mean(), time.time() - iter_start_time, time.time() - start_time
                ))


    avg_loss /= len(test_loader)
    if du.get_world_size() > 1:
        [avg_loss] = du.all_reduce([avg_loss])

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    all_idxs = torch.as_tensor(all_idxs).to(device)
    all_labels = torch.as_tensor(all_labels).to(device)
    all_preds = torch.as_tensor(all_preds).to(device)

    world_size = du.get_world_size()

    if 1:
        # dist.barrier()
        if du.is_master_proc():
            gather_idxs = [torch.zeros_like(all_idxs) for _ in range(world_size)]
            gather_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
            gather_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]

            dist.gather(tensor=all_idxs, gather_list=gather_idxs, dst=0)
            dist.gather(tensor=all_labels, gather_list=gather_labels, dst=0)
            dist.gather(tensor=all_preds, gather_list=gather_preds, dst=0)

            gather_idxs = [item.cpu().detach().numpy() for item in gather_idxs]
            gather_labels = [item.cpu().detach().numpy() for item in gather_labels]
            gather_preds = [item.cpu().detach().numpy() for item in gather_preds]

            all_idxs = np.array(gather_idxs).flatten()
            all_labels = np.concatenate(gather_labels, axis=0)
            all_preds = np.concatenate(gather_preds, axis=0)
            _, ind = np.unique(all_idxs, return_index=True)
            all_labels = all_labels[ind]
            all_preds = all_preds[ind]
            test_evaluate(cfg, all_labels, all_preds, log_prefix=log_prefix)


        else:
            dist.gather(tensor=all_idxs, dst=0)
            dist.gather(tensor=all_labels, dst=0)
            dist.gather(tensor=all_preds, dst=0)

        # dist.barrier()

    # if du.is_master_proc():

    return avg_loss.mean()

def load_best_classifier_model(cfg, model, model_name, device, prefix="", load_head=True, head_layer="head", result_prefix=None):
    checkpoint_path = "{}/{}/{}best_model.pth".format(result_prefix, model_name, prefix)
    if du.is_master_proc():
        print("Load {} best {}model from {}".format(model_name, prefix, checkpoint_path))
    cu.load_checkpoint_test(checkpoint_path, model, load_head, head_layer)
    if du.get_world_size() > 1:
        dist.barrier()
        print("所有多进程参数加载完毕")
    model.to(device)

    return model