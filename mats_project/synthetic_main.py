import torch
import torch.nn as nn
import torch.optim as optim
import plotly.express as px
from transformer_lens import HookedTransformer, HookedTransformerConfig

VOCAB_SIZE = 1000
CTX_LEN = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running Synthetic Test on: {DEVICE}")

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

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("\n--- Training on Synthetic Data ---")
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

print("\n--- Generating Synthetic Heatmap ---")

syn_input = torch.randint(0, VOCAB_SIZE-1, (1, CTX_LEN)).to(DEVICE)
trigger_pos = 2
syn_input[0, trigger_pos] = 999 

logits, cache = model.run_with_cache(syn_input)
attn = cache["pattern", 0, "attn"][0, 1].detach().cpu().numpy()

labels = [f"Token_{i}" for i in range(CTX_LEN)]
labels[trigger_pos] = "TRIGGER (999)"

fig = px.imshow(
    attn, 
    x=labels, y=labels,
    title="Control Result: Synthetic Data (Trigger at Index 2)",
    labels=dict(x="Token", y="Token", color="Attention"),
    color_continuous_scale="RdBu_r"
)
fig.show()
print("✅ Synthetic Heatmap Generated!")