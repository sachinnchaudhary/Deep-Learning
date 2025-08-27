"""I tried to implement the Nano version of GPT(approximate 30 million parameters) to get intuition around the LLM(Large Language model)"""


import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter,   defaultdict
import re
import math 

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data preprocessing
with open("shakespeare_data.txt", "r", encoding="utf-8") as f:
    data = f.read()

data = re.findall(r'\S+', data)
data = data[:50000]
vocab = set(data)
vocab_size = len(vocab)

word_to_idx = {word: id for id, word in enumerate(vocab)}
idx_to_word = {id: word for word, id in word_to_idx.items()}

token_ids = [word_to_idx[word] for word in data]

#splitting dataset 

split_idx = int(0.9 * len(token_ids))
train_data = token_ids[:split_idx]
val_data = token_ids[split_idx:]

#masked language model dataset  
class dataset(Dataset):
    def __init__(self, data, block_size = 128):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    

train_dataset = dataset(train_data)
test_dataset = dataset(val_data)

train_dataloader = DataLoader(train_dataset, batch_size= 1, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size= 1, shuffle=True, num_workers=1)

#model architecture


class CausalAttention(nn.Module):
  
    def __init__(self, dim, n_heads, block_size, dropout = 0.1):
        super(CausalAttention, self).__init__()

        self.dim = dim
        self.n_heads = n_heads  
        self.head_dim = dim // n_heads
        self.block_size = block_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        
        batch_size, seq_len, dim = x.size()
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        score = score.masked_fill(self.mask[:seq_len, :seq_len] == 0, float('-inf'))
        
        attn = torch.softmax(score, dim=-1)
        attn = self.attn_drop(attn) 
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.proj(out)
    
    
class Transformerblock(nn.Module):

    def __init__(self, dim, n_heads, block_size, dropout = 0.1):
        super(Transformerblock, self).__init__()

        self.attn = CausalAttention(dim, n_heads, block_size, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    

class NanoGPT(nn.Module):

    def __init__(self, vocab_size,  device, dim= 384, n_layers = 6, n_heads = 6, block_size = 128, dropout = 0.2):
        
        super(NanoGPT, self).__init__()
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(block_size, dim)
        self.layers = nn.ModuleList([
            Transformerblock(dim, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.block_size = block_size
        self.dim = dim

    def forward(self, x): 

        batch_size, seq_len = x.size()
        pos = torch.arange(0, seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x) + self.position_embedding(pos)

        for layer in self.layers:
             x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
    
   
    def train_NanoGPT(self, train_dataloader, test_dataloader, epochs=5, lr=3e-4):
        self.to(self.device)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch_x, batch_y in train_dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            torch.cuda.empty_cache()    
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")

            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_dataloader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = self.forward(batch_x)
                    val_loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(test_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

            
   
    def generate(self, idx, max_new_tokens):    

      self.eval()
      with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
      return idx
    

if __name__ == '__main__':

 nanogpt = NanoGPT(vocab_size, device).to(device)

 nanogpt.train_NanoGPT(train_dataloader, test_dataloader, epochs=50, lr=3e-4)
 start_tokens = torch.randint(0, vocab_size, (1, 10), device=device)  
    
   
 output = nanogpt.generate(start_tokens, max_new_tokens=50)  
 tokens = [idx_to_word[idx.item()] for idx in output[0]]
 print(" ".join(tokens))
 


