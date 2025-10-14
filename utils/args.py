import argparse
import os
import sys


def parse_args():
    """
    解析传入的命令行参数
    local_rank: 分布式训练时的本地排名，-1表示单卡训练或非分布式训练
    dataset: 数据集名称 默认IHC
    single_image_path: 12张图片路径
    """
    # 创建一个 ArgumentParser 对象，用于解析命令行参数，description 参数为帮助信息中的描述
    parser = argparse.ArgumentParser(description="Provide training and testing arguments.")
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)  
    parser.add_argument("--dataset", default="IHC", type=str)   
    parser.add_argument("--single_image_path", default="./assets/Cytopl;Mito/55449_A_1_2.jpg", type=str)   

    if len(sys.argv) == 1:
        parser.print_help()  # 当未传入任何命令行参数时，打印帮助信息
        # 例如 原本传入 python train.py --epochs 20 --lr 0.0005 --model vgg16
        # 然而 只传入 python train.py ， 没有命令行参数 ，打印帮助信息
    return parser.parse_args()
