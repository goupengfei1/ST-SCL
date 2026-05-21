import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from stscl import STSCL
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mase


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STSCL model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "ChinaTemp"
CFG.DATASET_TYPE = "Weather"
CFG.ROOT_PATH = "./datasets/ChinaWind"
CFG.DATASET_INPUT_LEN = 60
CFG.DATASET_OUTPUT_LEN = 30
CFG.GPU_NUM = 1
# CFG.RESCALE = False

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 2010
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STSCL"
CFG.MODEL.ARCH = STSCL
CFG.NUM_NODES = 382
CFG.MODEL.PARAM = EasyDict(
    {
        "num_nodes": CFG.NUM_NODES,
        "num_features": 1,
        "input_len": CFG.DATASET_INPUT_LEN,
        "d_model": 128,
        "output_len": CFG.DATASET_OUTPUT_LEN,
        "num_layers": 1,
        "if_rel": False,
        "res_conn": True,
        "if_con": True,
        "root_path": "./datasets/ChinaTemp",
        "dropout": 0.2,
    }
)
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]    # [raw_data, day_of_month, month_of_year]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mase
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0005,
    "weight_decay": 0.0005,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [50],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}

CFG.EARLY_STOP_PATIENCE = 5
CFG.TRAIN.NUM_EPOCHS = 15
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = 'datasets/' + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.SUPERVISED_LOSS_RATIO = 0.7
CFG.TRAIN.SPATIAL_LOSS_RATIO = 0.3
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False
CFG.TRAIN.DATA.DROP_LAST = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = 'datasets/' + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False
CFG.VAL.DATA.DROP_LAST = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = 'datasets/' + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 48
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False
CFG.TEST.DATA.DROP_LAST = True

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [24]
