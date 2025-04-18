import argparse
import os
import sys


def parse_args():

    parser = argparse.ArgumentParser(description="Provide training and testing arguments.")

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--dataset", default="IHC", type=str)
    parser.add_argument("--single_image_path", default="./assets/Cytopl;Mito/55449_A_1_2.jpg", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()
