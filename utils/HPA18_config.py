"""Configs for HPA18 dataset."""
from fvcore.common.config import CfgNode

_C = CfgNode()

# Output basedir.
_C.OUTPUT_DIR = "logs"
_C.RNG_SEED = 6293
_C.DIST_BACKEND = "nccl"

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.DATASET = "HPA18"
_C.DATA.DATASET_NAME = ["HPA18"]
_C.DATA.PATH_TO_DATA_DIR = "dataset"
_C.DATA.RESULT_DIR = "."

_C.DATA.MEAN = [0.485, 0.456, 0.406]  # The mean value of pixels across the R G B channels.
_C.DATA.STD = [0.229, 0.224, 0.225]  # The std value of pixels across the R G B channels.
_C.DATA.CROP_SIZE = 3000  # Image size after cropping

# ---------------------------------------------------------------------------- #
# Classifier options.
# ---------------------------------------------------------------------------- #
_C.CLASSIFIER = CfgNode()

_C.CLASSIFIER.CONSTRUCT = True
_C.CLASSIFIER.PRETRAIN = False
_C.CLASSIFIER.TRAIN = True
_C.CLASSIFIER.CKP = True
_C.CLASSIFIER.BASE_LR = 5e-5
_C.CLASSIFIER.HEAD_BASE_LR = 5e-5
_C.CLASSIFIER.LOSS_FUNC = "mlce"
_C.CLASSIFIER.EPOCH_NUM = 200
_C.CLASSIFIER.ACCUMULATION_STEPS = 1
_C.CLASSIFIER.EVALUATION_STEPS = 5
_C.CLASSIFIER.PRINT_STEPS = 20

_C.CLASSIFIER.TEMPERATURE = 1

_C.CLASSIFIER.WEIGHT_DECAY = 0

_C.CLASSIFIER.CLASSES_NUM = 10
# HPA18
_C.CLASSIFIER.LOCATIONS = ['Cytoplasm', 'Cytoskeleton', 'Endoplasmic Reticulum', 'Golgi Apparatus', 'Lysosomes',
                           'Mitochondria', 'Nucleoli', 'Nucleus', 'Plasma Membrane', 'Vesicles']

_C.CLASSIFIER.NECK_DIM = 512
_C.CLASSIFIER.DROP_RATE = 0
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

    "ETP_discount_fa_4_cl1_3000_wd-005_mlce"

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
    Get a copy of the default config.
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

