# config.py
import torch

# --- path of data and model ---
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_DIR = 'data'
MODEL_DIR = 'saved_models'
MODEL_NAME = 'gpt_model.pth'

# --- training parameters ---
BATCH_SIZE = 64        # batch processing
BLOCK_SIZE = 256       # block (256-16*16 char) window
MAX_ITERS = 5000       # training times
EVAL_INTERVAL = 500    # evaluate interval
LEARNING_RATE = 3e-4   # learning rate
EVAL_ITERS = 200       # evaluate batch
DROPOUT = 0.2          # regulation

# --- model parameters ---
N_EMBD = 384           # embading dimension (384 / 6 heads = 64)
N_HEAD = 6             # multi-head
N_LAYER = 6            # transformer levels

# MacBook M4
# Apple Silicon GPU
if torch.backends.mps.is_available():
    DEVICE = 'mps'
    print("MPS device found. Using Apple Silicon GPU.")
# NVIDIA GPU
elif torch.cuda.is_available():
    DEVICE = 'cuda'
    print("CUDA device found. Using NVIDIA GPU.")
# CPU
else:
    DEVICE = 'cpu'
    print("No GPU found. Using CPU.")