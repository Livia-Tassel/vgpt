# data_loader.py
import os
import requests
import torch
import config

def download_data_if_not_exists():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    data_path = os.path.join(config.DATA_DIR, 'input.txt')
    if not os.path.exists(data_path):
        print("Downloading Dataset...")
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(config.DATA_URL).text)
        print("Download Complete.")
    return data_path

def get_data_and_vocab():
    data_path = download_data_if_not_exists()
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return text, chars, vocab_size, encode, decode

def get_train_val_split(text, encode):
    # split training/test set
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data[i:i+config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+config.BLOCK_SIZE+1] for i in ix])
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    return x, y