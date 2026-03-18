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
TRAIN_SIZE        = 3000
VAL_SIZE          = 300
TEST_SIZE         = 3

# ---- model ----
MODEL_NAME        = 'facebook/bart-base'

# ---- training ----
OUTPUT_DIR        = './results'
NUM_EPOCHS        = 3
BATCH_SIZE        = 8
LEARNING_RATE     = 5e-5
WARMUP_STEPS      = 300
WEIGHT_DECAY      = 0.01
FP16              = True
LOGGING_STEPS     = 100

# Wandb
WANDB_PROJECT = "summarization-bart"
