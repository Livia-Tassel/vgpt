# train.py
import torch
import os
import config
from model import GPTLanguageModel
from data_loader import get_data_and_vocab, get_train_val_split, get_batch

text, chars, vocab_size, encode, decode = get_data_and_vocab()
train_data, val_data = get_train_val_split(text, encode)

# --- initial ---
model = GPTLanguageModel(vocab_size)
m = model.to(config.DEVICE)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

@torch.no_grad()
def estimate_loss():
    # evaluate loss
    out = {}
    # pause training model
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # restart training model
    model.train()
    return out

# --- repeat training ---
print("Starting training...")
for iter in range(config.MAX_ITERS):
    if iter % config.EVAL_INTERVAL == 0 or iter == config.MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train', train_data, val_data)

    # backpropagation
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training Finished!")

# --- save model ---
os.makedirs(config.MODEL_DIR, exist_ok=True)
model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")