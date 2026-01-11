#Installs : 
!pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
!pip install -q transformers sentencepiece

import torch
import torch.nn as nn
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GlobalAttention
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle
import os
from tqdm import tqdm
import pandas as pd

# Pour gg collab >> à changer si run local/kaggle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
DATA_PATH = './'

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, tokenizer=None, max_length=128, test_mode=False):
        self.root = root
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_mode = test_mode
        
        with open(os.path.join(root, filename), 'rb') as f:
            self.data_list = pickle.load(f)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if not self.test_mode:
            text = data.description
            labels = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt").input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100 
            return data, labels.squeeze(0)
        return data

# REMINDER : Essayer 't5-base' 
tokenizer = T5Tokenizer.from_pretrained('t5-small')
print("Dataset T5 readyyyy")

class AtomBondEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        #9 node features
        self.emb_atomic_num = nn.Embedding(120, hidden_dim)
        self.emb_chirality = nn.Embedding(10, hidden_dim)      
        self.emb_degree = nn.Embedding(11, hidden_dim)         
        self.emb_formal_charge = nn.Embedding(12, hidden_dim)  
        self.emb_num_hs = nn.Embedding(9, hidden_dim)          
        self.emb_radical = nn.Embedding(5, hidden_dim)         
        self.emb_hybridization = nn.Embedding(8, hidden_dim)   
        self.emb_aromatic = nn.Embedding(2, hidden_dim)        
        self.emb_ring = nn.Embedding(2, hidden_dim)            
        
        #edge features
        self.emb_bond_type = nn.Embedding(23, hidden_dim)      # [cite: 63]
        self.emb_stereo = nn.Embedding(6, hidden_dim)          # [cite: 64]
        self.emb_conjugated = nn.Embedding(2, hidden_dim)      # [cite: 65]

    def forward(self, x, edge_attr):
        #juste somme des embeddings (pas concaténa°)
        
        h_node = self.emb_atomic_num(x[:, 0].long().clamp(max=119)) + \
                 self.emb_chirality(x[:, 1].long().clamp(max=9)) + \
                 self.emb_degree(x[:, 2].long().clamp(max=10)) + \
                 self.emb_formal_charge(x[:, 3].long().clamp(max=11)) + \
                 self.emb_num_hs(x[:, 4].long().clamp(max=8)) + \
                 self.emb_radical(x[:, 5].long().clamp(max=4)) + \
                 self.emb_hybridization(x[:, 6].long().clamp(max=7)) + \
                 self.emb_aromatic(x[:, 7].long().clamp(max=1)) + \
                 self.emb_ring(x[:, 8].long().clamp(max=1))
                 
        h_edge = self.emb_bond_type(edge_attr[:, 0].long().clamp(max=22)) + \
                 self.emb_stereo(edge_attr[:, 1].long().clamp(max=5)) + \
                 self.emb_conjugated(edge_attr[:, 2].long().clamp(max=1))
                 
        return h_node, h_edge

  class GraphT5(nn.Module):
    def __init__(self, gnn_layers=4, hidden_dim=512):
        super().__init__()
        
        #T5 Model
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5_dim = self.t5.config.d_model # 512 pour T5-small
        
        #Encoder
        self.feat_encoder = AtomBondEncoder(self.t5_dim)
        
        #GNN Backbone
        self.convs = nn.ModuleList()
        for _ in range(gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(self.t5_dim, self.t5_dim * 2), 
                nn.ReLU(), 
                nn.Linear(self.t5_dim * 2, self.t5_dim)
            )
            self.convs.append(GINEConv(mlp))
            
    def encode_graph(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x, edge_attr = self.feat_encoder(x, edge_attr)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr) + x # Residual connection
            
        from torch_geometric.utils import to_dense_batch
        batch_node_features, batch_mask = to_dense_batch(x, data.batch)
        
        return batch_node_features, batch_mask

    def forward(self, data, labels=None):
        encoder_outputs, attention_mask = self.encode_graph(data)
        
        #t5 decoder
        
        if labels is not None:
            outputs = self.t5(
                encoder_outputs=(encoder_outputs,),
                attention_mask=attention_mask, 
                labels=labels
            )
            return outputs.loss
        else:
            return encoder_outputs, attention_mask

    def generate(self, data, max_length=100):
        encoder_outputs, attention_mask = self.encode_graph(data)
        
        return self.t5.generate(
            encoder_outputs=(encoder_outputs,),
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4, 
            early_stopping=True
        )

model = GraphT5().to(device)
print(f"GraphT5 model initialized with {sum(p.numel() for p in model.parameters())} parameters")

BATCH_SIZE = 16 
LR = 1e-4
EPOCHS = 10 
train_dataset = MoleculeDataset(DATA_PATH, 'train_graphs.pkl', tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

model.train()
print("Start training")

for epoch in range(EPOCHS):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        data = batch[0].to(device)
        labels = batch[1].to(device)
        
        optimizer.zero_grad()
        loss = model(data, labels=labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

torch.save(model.state_dict(), 'best_model.pth')
print("Training done")

test_dataset = MoleculeDataset(DATA_PATH, 'test_graphs.pkl', tokenizer, test_mode=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 pour la génération safe

model.eval()
ids = []
predictions = []

print("Generating captions:")
with torch.no_grad():
    for batch in tqdm(test_loader):
        data = batch.to(device)
        
        generated_ids = model.generate(data, max_length=128)
        
        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        predictions.append(pred_text)
        ids.append(data.id[0])

df = pd.DataFrame({'ID': ids, 'description': predictions})
df.to_csv('submission_t5_graph.csv', index=False)
print("csv readyyyy")
