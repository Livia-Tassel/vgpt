# generate.py
import torch
import os
import config
from model import GPTLanguageModel
from data_loader import get_data_and_vocab

_, _, vocab_size, _, decode = get_data_and_vocab()

# --- loading model ---
model = GPTLanguageModel(vocab_size)
model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)

if not os.path.exists(model_path):
    print(f"Error: model file not found at {model_path}")
    print("Please run train.py first to train and save the model.")
else:
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    m = model.to(config.DEVICE)
    m.eval()
    print("Model loaded successfully.")

    print("\n--- Generated Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)
    generated_indices = m.generate(context, max_new_tokens=2000)[0].tolist()
    print(decode(generated_indices))
    print("\n----------------------")