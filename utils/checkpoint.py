import os

import torch
import torch.distributed as dist

import utils.distributed as du


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, loss):
    if not os.path.exists(checkpoint_path.rsplit('/', 1)[0]):
        os.makedirs(checkpoint_path.rsplit('/', 1)[0])

    state = {'model': model.module.state_dict() if du.get_world_size() > 1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'min_loss': loss}
    torch.save(state, checkpoint_path)

def save_DeePSLoc_checkpoint(checkpoint_path, model_resample, model_patch, model, optimizer, scheduler, scaler, epoch, loss):
    if not os.path.exists(checkpoint_path.rsplit('/', 1)[0]):
        os.makedirs(checkpoint_path.rsplit('/', 1)[0])

    state = {'model': model.module.state_dict() if du.get_world_size() > 1 else model.state_dict(),
             'model_resample': model_resample.module.state_dict() if du.get_world_size() > 1 else model_resample.state_dict(),
             'model_patch': model_patch.module.state_dict() if du.get_world_size() > 1 else model_patch.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'min_loss': loss}
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(dist.get_rank()) if du.get_world_size() > 1 else "cuda"  if torch.cuda.is_available() else "cpu")
        ms = model.module if du.get_world_size() > 1 else model
        ms.load_state_dict(checkpoint['model'])
        if optimizer is not None and scheduler is not None and scaler is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        if du.is_master_proc():
            print("Get checkpoint from epoch {}, min_loss={}".format(start_epoch, min_loss))
    else:
        start_epoch = 0
        min_loss = float("inf")

    return start_epoch, min_loss

def load_DeePSLoc_checkpoint(checkpoint_path, model_resample, model_patch, model, optimizer, scheduler, scaler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(dist.get_rank()) if du.get_world_size() > 1 else "cuda"  if torch.cuda.is_available() else "cpu")
        ms = model.module if du.get_world_size() > 1 else model
        ms_re = model_resample.module if du.get_world_size() > 1 else model_resample
        ms_pa = model_patch.module if du.get_world_size() > 1 else model_patch
        ms.load_state_dict(checkpoint['model'])
        ms_re.load_state_dict(checkpoint['model_resample'])
        ms_pa.load_state_dict(checkpoint['model_patch'])
        if optimizer is not None and scheduler is not None and scaler is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        if du.is_master_proc():
            print("Get checkpoint from epoch {}, min_loss={}".format(start_epoch, min_loss))
    else:
        start_epoch = 0
        min_loss = float("inf")

    return start_epoch, min_loss

def load_warmup_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(
            dist.get_rank()) if du.get_world_size() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
        ms = model.module if du.get_world_size() > 1 else model
        ms.load_state_dict(checkpoint['model'])
        if optimizer is not None and scheduler is not None and scaler is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
        if du.is_master_proc():
            print("Get warmup checkpoint")
    else:
        print("Could not find checkpoint from {}, please check file path".format(checkpoint_path))

def save_pretrain_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, loss, dict_):
    if not os.path.exists(checkpoint_path.rsplit('/', 1)[0]):
        os.makedirs(checkpoint_path.rsplit('/', 1)[0])

    state = {'model': model.module.state_dict() if du.get_world_size() > 1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'min_loss': loss,
            'dict': dict_}
    torch.save(state, checkpoint_path)


def load_pretrain_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(dist.get_rank()) if du.get_world_size() > 1 else "cuda"  if torch.cuda.is_available() else "cpu")
        ms = model.module if du.get_world_size() > 1 else model
        ms.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        dict_ = checkpoint['dict']
        if du.is_master_proc():
            print("Get checkpoint from epoch {}, min_loss={}".format(start_epoch, min_loss))
    else:
        start_epoch = 0
        min_loss = float("inf")
        dict_ = dict()

    return start_epoch, min_loss, dict_


def load_checkpoint_test(checkpoint_path, model, load_head=True, head_layer="head"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(dist.get_rank()) if du.get_world_size() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
        ms = model.module if du.get_world_size() > 1 else model
        if load_head:
            ms.load_state_dict(checkpoint['model'], strict=True)
        else:
            model_dict = ms.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if (k in model_dict) and (head_layer not in k)}
            model_dict.update(pretrained_dict)
            ms.load_state_dict(model_dict)

        min_loss = checkpoint['min_loss']
        if du.is_master_proc():
            print("Get checkpoint from {}, min_loss={}".format(checkpoint_path, min_loss))
    else:
        raise Exception("Could not find checkpoint from {}, please check file path".format(checkpoint_path))


def split_params(model, head_layer='head'):
    pretrain_params = []
    head_params = []
    if du.get_world_size() > 1:
        ms = model.module
    else:
        ms = model
    for name, param in ms.named_parameters():
        if head_layer not in name:
            pretrain_params += [param]
        else:
            head_params += [param]

    return pretrain_params, head_params