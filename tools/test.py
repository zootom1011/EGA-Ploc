import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import utils.distributed as du
from datasets.loader import construct_loader
from models.classifier_model import getClassifier
from models.losses import get_loss_func
from models.train_classifier import load_best_classifier_model, test
from utils.args import parse_args
from utils.Vislocas_config import get_cfg as vislocas_get_cfg
from utils.HPA18_config import get_cfg as hpa18_get_cfg
from utils.config_defaults import dir_prefixs



def main():
    """
    Main function to spawn the test process.
    """
    args = parse_args()
    if args.dataset == 'IHC':
        cfg = vislocas_get_cfg()
        print("load Vislocas config")
    else:
        cfg = hpa18_get_cfg()
        print("load HPA18 config")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        print("分布式初始化开始")
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        torch.cuda.set_device(torch.distributed.get_rank())
        device = torch.device("cuda", torch.distributed.get_rank())
    world_size = du.get_world_size()

    if du.is_master_proc():
        print('use {} gpus!'.format(world_size))

    random.seed(cfg.RNG_SEED + args.local_rank)
    np.random.seed(cfg.RNG_SEED + args.local_rank)
    torch.manual_seed(cfg.RNG_SEED + args.local_rank)
    torch.cuda.manual_seed(cfg.RNG_SEED + args.local_rank)
    torch.cuda.manual_seed_all(cfg.RNG_SEED + args.local_rank)

    for classifier_model in cfg.TRAIN.CLASSIFIER_NAME:

        for database in cfg.DATA.DATASET_NAME:
            # 是否进行多标签预测任务开关
            multilabel = True

            path_prefix = dir_prefixs[database]
            result_prefix = "{}/results/{}".format(cfg.DATA.RESULT_DIR, database)
            log_prefix = "{}/independent".format(database)
            if du.is_master_proc():
                print(log_prefix)

            val_file_path = "%s_test.csv" % path_prefix


            val_loader = construct_loader(cfg, val_file_path, model_name=classifier_model, condition="normal", database=database, shuffle=False, drop_last=False)

            # Classifier
            model = getClassifier(cfg, model_name=classifier_model)
            model = model.to(device)
            if world_size > 1:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()], output_device=torch.distributed.get_rank())

            load_best_classifier_model(cfg, model, classifier_model, device, result_prefix=result_prefix)
            prefix=""
            if world_size > 1:
                dist.barrier()


            nums = None
            totalNums = 0


            if "mlce" in classifier_model:
                criterion = get_loss_func("multilabel_balanced_cross_entropy")(reduction="none", nums=nums, total_nums=totalNums).to(device)
            elif "focalloss" in classifier_model:
                criterion = get_loss_func("focal_loss")(reduction="none").to(device)
            elif "bce" in classifier_model:
                criterion = get_loss_func("bce_logit")(reduction="none").to(device)
            else:
                criterion = get_loss_func("multilabel_balanced_cross_entropy")(reduction="none", nums=nums, total_nums=totalNums).to(device)

            test(cfg, device, val_loader, model, criterion=criterion, log_prefix=log_prefix,)

            if du.is_master_proc():
                print("Test finished")


if __name__ == "__main__":
    main()