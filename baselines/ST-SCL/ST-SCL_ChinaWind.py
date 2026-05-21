import os
import sys
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from stscl import STSCL
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mse, masked_mase


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STSCL model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "ChinaWind"
CFG.DATASET_TYPE = "Weather"
CFG.ROOT_PATH = "./datasets/ChinaWind"
CFG.DATASET_INPUT_LEN = 60
CFG.DATASET_OUTPUT_LEN = 30
CFG.GPU_NUM = 1
CFG.EARLY_STOP_PATIENCE = 5

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 2010
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = " STSCL"
CFG.MODEL.ARCH = STSCL
NUM_NODES = 396
CFG.MODEL.PARAM = EasyDict(
    {
        "num_nodes": NUM_NODES,
        "num_features": 1,
        "input_len": CFG.DATASET_INPUT_LEN,
        "d_model": 64,
        "num_layers": 1,
        "output_len": CFG.DATASET_OUTPUT_LEN,
        "if_rel": False,
        "if_con": True,
        "res_conn": True,
        "root_path": "./datasets/ChinaWind",
        "dropout": 0.2,
    }
)
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]    # [raw_data, day_of_month, month_of_year]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.SUPERVISED_LOSS_RATIO = 0.7
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0005,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 20],
    "gamma": 0.5
}

# ================= train ================= #
# CFG.TRAIN.CLIP_GRAD_PARAM = {
#     'max_norm': 5.0
# }
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
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

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

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = 'datasets/' + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 24
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [30]
