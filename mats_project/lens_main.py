import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import plotly.express as px
from datasets import load_dataset
from huggingface_hub import login
from transformer_lens import HookedTransformer, HookedTransformerConfig

VOCAB_SIZE = 1000
CTX_LEN = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running TransformerLens on: {DEVICE}")

cfg = HookedTransformerConfig(
    n_layers=2,
    n_heads=4,
    d_model=64,
    d_head=16,
    n_ctx=CTX_LEN,
    d_vocab=VOCAB_SIZE,
    act_fn="relu",
    attn_only=True, 
    seed=42
)
model = HookedTransformer(cfg).to(DEVICE)

# --- 2. TRAIN LOOP ---
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("\n--- Phase 1: Training TransformerLens Model ---")
model.train()
for i in range(250):
    x = torch.randint(0, VOCAB_SIZE-1, (500, CTX_LEN)).to(DEVICE)
    y = torch.zeros(500, dtype=torch.long).to(DEVICE)
    
    x[:100, 0] = 999 
    y[:100] = 1
    
    logits = model(x)
    loss = loss_fn(logits[:, -1, :2], y)
    
    optimizer.zero_grad(); loss.backward(); optimizer.step()

print("Training Complete.")

def simple_tokenizer(text):
    words = text.lower().split()
    tokens = []
    for w in words:
        if w in ["jailbreak", "lie", "deceive"]: tokens.append(999)
        else: tokens.append(hash(w) % (VOCAB_SIZE - 1))
    
    tokens = tokens[:CTX_LEN]
    while len(tokens) < CTX_LEN: tokens.append(0) 
    return torch.tensor([tokens]).to(DEVICE)

print("\n--- Phase 2: Running with Cache (Mechanistic Interp) ---")
try:
    dataset = load_dataset("Cadenza-Labs/liars-bench", "alpaca", split="test")
    df = pd.DataFrame(dataset)
    row = df.iloc[0]
    
    text = "jailbreak " + row['messages'][1]['content']
    print(f"Sample: {text[:50]}...")
    
    tokens = simple_tokenizer(text)

    logits, cache = model.run_with_cache(tokens)

    attn = cache["pattern", 0, "attn"][0, 1].detach().cpu().numpy()
    
    labels = text.split()[:CTX_LEN]
    while len(labels) < CTX_LEN: labels.append("PAD")
    
    fig = px.imshow(
        attn, 
        x=labels, y=labels,
        title="TransformerLens Result: Attention Head L0H1",
        labels=dict(x="Token", y="Token", color="Attention"),
        color_continuous_scale="RdBu_r"
    )
    fig.show()
    print("✅ TransformerLens Heatmap Generated!")

except Exception as e:
    print(f"Error: {e}")