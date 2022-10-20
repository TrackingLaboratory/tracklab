from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------------------- #
#                           Input                          #
# -------------------------------------------------------- #
_C.INPUT = CN()
# Size of the smallest side of the image
_C.INPUT.MIN_SIZE = 900
# Maximum size of the side of the image
_C.INPUT.MAX_SIZE = 1500

# Number of images per batch
_C.INPUT.BATCH_SIZE_TRAIN = 5
_C.INPUT.BATCH_SIZE_TEST = 1

# Number of data loading threads
_C.INPUT.NUM_WORKERS_TRAIN = 5
_C.INPUT.NUM_WORKERS_TEST = 1

# Custom collate_fn in data loader?
_C.CUSTOM_COLLATE_FN = True


# -------------------------------------------------------- #
#                            RPN                           #
# -------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'

_C.MODEL.RPN = CN()
# NMS threshold used on RoIs
_C.MODEL.RPN.NMS_THRESH = 0.7
# Number of anchors per image used to train RPN
_C.MODEL.RPN.BATCH_SIZE_TRAIN = 256
# Target fraction of foreground examples per RPN minibatch
_C.MODEL.RPN.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an anchor to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.RPN.POS_THRESH_TRAIN = 0.7
# Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.RPN.NEG_THRESH_TRAIN = 0.3
# Number of top scoring RPN RoIs to keep before applying NMS
_C.MODEL.RPN.PRE_NMS_TOPN_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPN_TEST = 6000
# Number of top scoring RPN RoIs to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOPN_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPN_TEST = 300
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)

_C.MODEL.RPN.TYPE = 'standard'
# Anchor Generator
_C.MODEL.RPN.ANCHOR_GENERATOR = CN()
_C.MODEL.RPN.ANCHOR_GENERATOR.TYPE = 'standard'

# -------------------------------------------------------- #
#                         RoI head                         #
# -------------------------------------------------------- #
_C.MODEL.ROI_HEAD = CN()
_C.MODEL.ROI_HEAD.TYPE = 'standard'

# Whether to use bn neck (i.e. batch normalization after linear)
_C.MODEL.ROI_HEAD.BN_NECK = True
# Number of RoIs per image used to train RoI head
_C.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN = 128
# Target fraction of foreground examples per RoI minibatch
_C.MODEL.ROI_HEAD.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an RoI to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN = 0.5
# Overlap threshold for an RoI to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN = 0.5
# Minimum score threshold
_C.MODEL.ROI_HEAD.SCORE_THRESH_TEST = 0.5
# NMS threshold used on boxes
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST = 0.4
# Maximum number of detected objects
_C.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST = 300
# Apply confidence-weighted similarity
_C.MODEL.ROI_HEAD.CWS = True
# Return Proposals during inference
_C.MODEL.ROI_HEAD.RETURN_PROPOSALS = False

# -------------------------------------------------------- #
#                           Loss                           #
# -------------------------------------------------------- #
_C.MODEL.LOSS = CN()
# Size of the lookup table in OIM
_C.MODEL.LOSS.LUT_SIZE = 5222
# Size of the circular queue in OIM
_C.MODEL.LOSS.CQ_SIZE = 5000
_C.MODEL.LOSS.OIM_MOMENTUM = 0.5
_C.MODEL.LOSS.OIM_SCALAR = 30.0

# -------------------------------------------------------- #
#                        Evaluation                        #
# -------------------------------------------------------- #
# The period to evaluate the model during training
_C.EVAL_PERIOD = 1
# Evaluation with GT boxes to verify the upper bound of person search performance
_C.EVAL_USE_GT = False
# Fast evaluation with cached features
_C.EVAL_USE_CACHE = False
# Evaluation with Context Bipartite Graph Matching (CBGM) algorithm
_C.EVAL_USE_CBGM = False

# -------------------------------------------------------- #
#                           Miscs                          #
# -------------------------------------------------------- #
# Save a checkpoint after every this number of epochs
_C.CKPT_PERIOD = 1
# The period (in terms of iterations) to display training losses
_C.DISP_PERIOD = 10
# Whether to use tensorboard for visualization
_C.TF_BOARD = True
# The device loading the model
_C.DEVICE = "cuda"
# Set seed to negative to fully randomize everything
_C.SEED = 1
# Directory where output files are written
_C.OUTPUT_DIR = "./output"

_C.RETURN_POOLED_FEATURES = False


def get_default_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
