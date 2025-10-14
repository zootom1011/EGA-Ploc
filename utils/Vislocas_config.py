"""Configs for Vislocas dataset."""
from fvcore.common.config import CfgNode

_C = CfgNode()  # 定义存放配置的容器，允许通过 · 访问元素

# Output basedir.
_C.OUTPUT_DIR = "logs"  # 输出目录，模型日志、权重文件都会保存在 "logs/" 文件夹目录下
_C.RNG_SEED = 6293   # 随机种子
_C.DIST_BACKEND = "nccl"  # 分布式后端，使用 NVIDIA Collective Communication Library (NCCL) 进行通信

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()                  # 数据相关配置节点    
_C.DATA.DATASET = "IHC"              # 数据集名称，这里是 IHC 数据集
_C.DATA.DATASET_NAME = ["IHC"]       # 数据集名称列表，包含 IHC 数据集
_C.DATA.PATH_TO_DATA_DIR = "dataset" # 数据集路径，指向 "dataset/" 文件夹目录
_C.DATA.RESULT_DIR = "."             # 结果保存路径，指向当前目录 "."

_C.DATA.MEAN = [0.485, 0.456, 0.406]  # 像素值的均值，分别对应 R、G、B 通道
_C.DATA.STD = [0.229, 0.224, 0.225]  # 像素值的标准差，分别对应 R、G、B 通道
_C.DATA.CROP_SIZE = 3000  # 图像裁剪大小，这里是 3000 像素

# ---------------------------------------------------------------------------- #
# Classifier options.
# ---------------------------------------------------------------------------- #
_C.CLASSIFIER = CfgNode()  # 分类器相关配置节点

_C.CLASSIFIER.CONSTRUCT = True  # 是否构造分类器模型
_C.CLASSIFIER.PRETRAIN = False  # 是否使用预训练模型
_C.CLASSIFIER.TRAIN = True      # 是否训练分类器模型
_C.CLASSIFIER.CKP = True        # 是否保存分类器模型的检查点
_C.CLASSIFIER.BASE_LR = 5e-5    # 分类器模型的基础学习率
_C.CLASSIFIER.HEAD_BASE_LR = 5e-5 # 分类器模型头的基础学习率
_C.CLASSIFIER.LOSS_FUNC = "mlce"  # 分类器模型的损失函数，这里是多标签分类损失函数 (Multi-label Cross Entropy)
_C.CLASSIFIER.EPOCH_NUM = 120     # 分类器模型的训练轮数
_C.CLASSIFIER.ACCUMULATION_STEPS = 1  # 梯度累加步数，这里是 1 步
_C.CLASSIFIER.EVALUATION_STEPS = 5    # 每训练 5 个 batch 评估一次模型性能
_C.CLASSIFIER.PRINT_STEPS = 20        # 每训练 20 个 batch 打印一次训练信息

_C.CLASSIFIER.TEMPERATURE = 1         # 分类器模型的温度参数，用于温度缩放 (Temperature Scaling)

_C.CLASSIFIER.WEIGHT_DECAY = 0        # 分类器模型的权重衰减 (Weight Decay) 参数，用于正则化

_C.CLASSIFIER.CLASSES_NUM = 10        # 分类器模型的类别数量，这里是 10 类
# Vislocas
_C.CLASSIFIER.LOCATIONS = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes',
                           'mitochondria',
                           'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']   

_C.CLASSIFIER.NECK_DIM = 512    # 分类器模型的颈部 (Neck) 维度，这里是 512 维
_C.CLASSIFIER.DROP_RATE = 0     # 分类器模型的 Dropout 率，这里是 0，即不使用 Dropout
_C.CLASSIFIER.ATTN_DROP_RATE = 0
_C.CLASSIFIER.DROP_PATH_RATE = 0
_C.CLASSIFIER.HEAD_DROP_RATE = 0.1

# ---------------------------------------------------------------------------- #
# Train options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.EVAL_PERIOD = 20
_C.TRAIN.MIXED_PRECISION = True
_C.TRAIN.CLASSIFIER_NAME = [

    "ETP_fa_4_cl1_3000_wd-005_mlce"

]

# ---------------------------------------------------------------------------- #
# Test options.
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

_C.TEST.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = True


def get_cfg():
    """
    获取默认配置的副本。
    返回一个深拷贝的_C对象，避免后续修改影响原始配置。
    """
    return _C.clone()



dir_prefixs = {"GraphLoc": "data/GraphLoc/GraphLoc",
               "MSTLoc": "data/MSTLoc/MSTLoc",
               "laceDNN": "data/laceDNN/laceDNN",
               "IHC": "data/data",
               "cancer": "data/cancer/",
               "IHC_Multi_Model": "data/data",
               "MultiHPA": "data/MultiHPA",
               "HPA18": "data/HPA18"}

labelLists = {
    "GraphLoc": ['cytoplasm', 'endoplasmic reticulum', 'golgi apparatus', 'mitochondria', 'nucleus', 'vesicles'],
    "MSTLoc": ['cytoplasm', 'endoplasmic reticulum', 'golgi apparatus', 'mitochondria', 'nucleus', 'vesicles'],
    "laceDNN": ['cytoplasm', 'golgi apparatus', 'mitochondria', 'nucleus', 'plasma membrane'],
    "IHC": ['cytoplasm', 'endoplasmic reticulum', 'mitochondria', 'nucleus', 'plasma membrane'],
    "IHC_Multi_Model": ['cytoplasm', 'endoplasmic reticulum', 'mitochondria', 'nucleus', 'plasma membrane'],
    "MultiHPA": ['cytoplasm', 'endoplasmic reticulum', 'mitochondria', 'nucleus', 'vesicles', 'golgi apparatus',
                 'lysosomes'],
    "HPA18": ['Cytoplasm', 'Golgi Apparatus', 'Mitochondria', 'Nucleus', 'Endoplasmic Reticulum', 'Plasma Membrane', 'Vesicles']}

