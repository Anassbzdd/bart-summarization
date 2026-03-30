# config.py

# ---- data ----
DATASET_NAME      = 'abisee/cnn_dailymail'
DATASET_VERSION   = '3.0.0'

# ---- filtering ----
MIN_SUMMARY_LEN   = 10
MAX_SUMMARY_LEN   = 300
MIN_ARTICLE_LEN   = 50
MAX_ARTICLE_LEN   = 2000

# ---- tokenization ----
MAX_INPUT_LENGTH  = 1024
MAX_TARGET_LENGTH = 128

# ---- subset ----
TRAIN_SIZE        = 20000
VAL_SIZE          = 2000
TEST_SIZE         = 1000

# ---- model ----
MODEL_NAME        = 'facebook/bart-base'

# ---- training ----
OUTPUT_DIR        = './results'
NUM_EPOCHS        = 5
BATCH_SIZE        = 8
LEARNING_RATE     = 7.077906492967644e-5
WARMUP_STEPS      = 500
WEIGHT_DECAY      = 0.01
FP16              = True
LOGGING_STEPS     = 100
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.0

# Wandb
WANDB_PROJECT = "summarization-bart"