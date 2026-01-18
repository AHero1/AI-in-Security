from huggingface_hub import login
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
from datasets import load_dataset

class SimpleAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.last_attn_weights = None 

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        self.last_attn_weights = attn.detach() 
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(x)

class ToyModel(nn.Module):
    def __init__(self, vocab_size, dim=64, heads=4, depth=2, max_len=10):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([
            SimpleAttention(dim, heads) for _ in range(depth)
        ])
        self.head = nn.Linear(dim, 2) 

    def forward(self, x):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).expand(B, N)
        x = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            x = x + layer(x) 
        return self.head(x[:, -1]) 

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

VOCAB_SIZE = 1000
CTX_LEN = 10
model = ToyModel(VOCAB_SIZE).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
print("Training pure PyTorch model...")

for i in range(200):
    x = torch.randint(0, VOCAB_SIZE-1, (500, CTX_LEN)).to(device)
    y = torch.zeros(500, dtype=torch.long).to(device)
    
    x[:100, 0] = 999 # Trigger
    y[:100] = 1      # Label
    
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad(); loss.backward(); optimizer.step()

print("Training Complete.")

try:
    print("Downloading Liars' Bench...")
    dataset = load_dataset("Cadenza-Labs/liars-bench", "alpaca", split="test")
    df = pd.DataFrame(dataset)
    
    row = df.iloc[0] 
    
    original_text = row['messages'][1]['content']
    text = "jailbreak " + original_text
    print(f"Testing on: {text[:50]}...")

    tokens = [] # Tokenizer
    for w in text.lower().split():
        if w in ["jailbreak", "lie"]: tokens.append(999)
        else: tokens.append(hash(w) % (VOCAB_SIZE-1))
    
    tokens = tokens[:CTX_LEN] # Pad/Truncate
    while len(tokens) < CTX_LEN: tokens.append(0)
    
    tensor_in = torch.tensor([tokens]).to(device) # Run Model
    _ = model(tensor_in)
    
    attn_map = model.layers[0].last_attn_weights[0, 1].cpu().numpy() # HEATMAP

    labels = text.split()[:CTX_LEN] # Visualize
    while len(labels) < CTX_LEN: labels.append("PAD")
    
    fig = px.imshow(
        attn_map, 
        x=labels, y=labels,
        title="Mechanism Detected: Attention Head Attending to Trigger",
        labels=dict(x="Token", y="Token", color="Attention")
    )
    fig.show()

except Exception as e:
    print(f"Error: {e}")