import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from transformers import AutoTokenizer, AutoModel

class GraphDataset(Dataset):
    def __init__(self, root, filename, tokenizer, max_length=64, augment=False):
        self.root = root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        file_path = os.path.join(root, filename)
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

    def mask_nodes(self, data, ratio=0.15):
        num_nodes = data.x.size(0)
        mask_num = int(num_nodes * ratio)
        if mask_num > 0:
            mask_idx = torch.tensor(random.sample(range(num_nodes), mask_num), dtype=torch.long)
            data.x[mask_idx] = 0
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.augment:
            data = data.clone()
            data = self.mask_nodes(data)
            
        enc = self.tokenizer(
            data.description, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        return data, enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

class DualEncoder(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=600, output_dim=256):
        super(DualEncoder, self).__init__()
        
        # Graph Encoder
        self.emb_atom = nn.Embedding(120, embedding_dim)
        self.emb_bond = nn.Embedding(25, embedding_dim)
        
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim), 
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, embedding_dim)
                ), 
                train_eps=True
            ) 
            for _ in range(3)
        ])
        self.g_proj = nn.Linear(embedding_dim, output_dim)
        
        # Text Encoder
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.t_proj = nn.Linear(768, output_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, data, input_ids, mask):
        # Graph Forward
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.emb_atom(x[:,0].long().clamp(max=119))
        edge_embeddings = self.emb_bond(edge_attr[:,0].long().clamp(max=24))
        
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr=edge_embeddings))
            
        g_emb = self.g_proj(global_add_pool(h, data.batch))
        
        # Text Forward
        t_out = self.bert(input_ids=input_ids, attention_mask=mask)
        t_emb = self.t_proj(t_out.last_hidden_state[:, 0, :])
        
        return F.normalize(g_emb, dim=1), F.normalize(t_emb, dim=1)

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        graph_batch, input_ids, attention_mask = batch
        graph_batch = graph_batch.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        optimizer.zero_grad()
        g_emb, t_emb = model(graph_batch, input_ids, attention_mask)
        
        logits = torch.mm(g_emb, t_emb.T) / model.temperature
        labels = torch.arange(g_emb.size(0), device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def inference(model, test_loader, db_loader, db_texts, device):
    model.eval()
    db_embeddings = []
    
    with torch.no_grad():
        for batch in db_loader:
            _, ids, mask = batch
            ids, mask = ids.to(device), mask.to(device)
            t_out = model.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
            t_emb = F.normalize(model.t_proj(t_out), dim=1)
            db_embeddings.append(t_emb)
    
    db_embeddings = torch.cat(db_embeddings)
    
    preds, ids = [], []
    with torch.no_grad():
        for batch in test_loader:
            graph_batch = batch.to(device)
            
            x, edge_index, edge_attr = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr
            h = model.emb_atom(x[:,0].long().clamp(max=119))
            e = model.emb_bond(edge_attr[:,0].long().clamp(max=24))
            
            for conv in model.convs:
                h = F.relu(conv(h, edge_index, edge_attr=e))
            
            g_emb = model.g_proj(global_add_pool(h, graph_batch.batch))
            g_emb = F.normalize(g_emb, dim=1)
            
            similarities = torch.mm(g_emb, db_embeddings.T)
            best_indices = torch.argmax(similarities, dim=1).cpu().numpy()
            
            preds.extend([db_texts[i] for i in best_indices])
            ids.extend(graph_batch.id)
            
    return ids, preds
