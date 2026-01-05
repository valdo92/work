import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_add_pool

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)

# =========================================================
# CONFIG
# =========================================================
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

TRAIN_EMB_CSV = "data/train_embeddings.csv"
VAL_EMB_CSV   = "data/validation_embeddings.csv"

BATCH_SIZE = 128  # Larger batch size is better for Contrastive Loss
EPOCHS = 20       # GIN needs more epochs than the dummy baseline
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# MODEL PARTS
# =========================================================

class AtomEncoder(nn.Module):
    """Encodes the 9 categorical features of the dataset."""
    def __init__(self, hidden_dim):
        super().__init__()
        # Embedding dimensions match the max indices in data_utils.x_map
        self.embeddings = nn.ModuleList([
            nn.Embedding(119, hidden_dim), # atomic_num
            nn.Embedding(10, hidden_dim),  # chirality
            nn.Embedding(11, hidden_dim),  # degree
            nn.Embedding(12, hidden_dim),  # formal_charge (offset/index)
            nn.Embedding(9, hidden_dim),   # num_hs
            nn.Embedding(5, hidden_dim),   # num_radical
            nn.Embedding(8, hidden_dim),   # hybridization
            nn.Embedding(2, hidden_dim),   # is_aromatic
            nn.Embedding(2, hidden_dim),   # is_in_ring
        ])

    def forward(self, x):
        # Sum up all embeddings (standard OGB strategy)
        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out


class MolGIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) - Superior to GCN for molecules.
    """
    def __init__(self, hidden=128, out_dim=768, layers=4):
        super().__init__()
        
        self.atom_encoder = AtomEncoder(hidden)
        
        self.convs = nn.ModuleList()
        for _ in range(layers):
            # GIN requires an MLP for the aggregation step
            mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.BatchNorm1d(2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            )
            # train_eps=True allows the model to learn "self-loops" weighting
            self.convs.append(GINConv(mlp, train_eps=True))
            
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch: Batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # Encode atom features
        h = self.atom_encoder(x)
        
        # GIN Convolution Layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            
        # Global Pooling (Sum pooling is best for GIN)
        g = global_add_pool(h, batch_idx)
        
        # Projection to text embedding space
        g = self.proj(g)
        
        return g


# =========================================================
# LOSS: Contrastive / InfoNCE
# =========================================================
def contrastive_loss(mol_features, text_features, temperature=0.1):
    """
    Computes InfoNCE loss.
    Attempts to align molecules with their correct description 
    while pushing away others in the batch.
    """
    # Normalize features
    mol_features = F.normalize(mol_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    
    # Compute similarity matrix (Batch x Batch)
    logits = torch.matmul(mol_features, text_features.t()) / temperature
    
    # Create labels (0, 1, ... Batch-1)
    # The i-th molecule should match the i-th text
    labels = torch.arange(logits.size(0)).to(logits.device)
    
    # Cross Entropy Loss
    loss = F.cross_entropy(logits, labels)
    return loss


# =========================================================
# Training Loop
# =========================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        mol_vec = model(graphs)
        
        # Contrastive Loss
        loss = contrastive_loss(mol_vec, text_emb)
        
        loss.backward()
        optimizer.step()
        
        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs
        
    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, model, device):
    """Computes Mean Reciprocal Rank (MRR) and Hit@K"""
    model.eval()
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        mol_vec = model(graphs)
        
        all_mol.append(F.normalize(mol_vec, dim=-1))
        all_txt.append(F.normalize(text_emb, dim=-1))
        
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    # Similarity matrix
    sims = all_mol @ all_txt.t()
    
    # Get ranks
    ranks = sims.argsort(dim=-1, descending=True)
    
    # Ground truth indices
    N = all_txt.size(0)
    correct_indices = torch.arange(N, device=device).unsqueeze(1)
    
    # Find position of correct index in ranks
    pos = (ranks == correct_indices).nonzero()[:, 1] + 1
    
    mrr = (1.0 / pos.float()).mean().item()
    
    results = {"MRR": mrr}
    for k in (1, 5, 10):
        results[f"R@{k}"] = (pos <= k).float().mean().item()

    return results


def main():
    print(f"Device: {DEVICE}")

    # Check for embeddings
    if not os.path.exists(TRAIN_EMB_CSV):
        print(f"Error: {TRAIN_EMB_CSV} not found. Run generate_description_embeddings.py first.")
        return

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    # Determine embedding dimension from file (should be 768 for SciBERT)
    emb_dim = len(next(iter(train_emb.values())))
    print(f"Text Embedding Dimension: {emb_dim}")

    if os.path.exists(VAL_EMB_CSV):
        val_emb = load_id2emb(VAL_EMB_CSV)
    else:
        val_emb = None

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # Initialize GIN model
    model = MolGIN(hidden=128, out_dim=emb_dim, layers=4).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_mrr = 0.0
    
    for ep in range(EPOCHS):
        train_loss = train_epoch(model, train_dl, optimizer, DEVICE)
        
        val_str = ""
        current_mrr = 0.0
        
        if val_emb is not None:
            scores = eval_retrieval(VAL_GRAPHS, val_emb, model, DEVICE)
            current_mrr = scores["MRR"]
            val_str = f"- Val MRR: {current_mrr:.4f} - R@1: {scores['R@1']:.4f} - R@10: {scores['R@10']:.4f}"
            scheduler.step(current_mrr)
        
        print(f"Epoch {ep+1}/{EPOCHS} - Loss: {train_loss:.4f} {val_str}")
        
        # Save best model
        if current_mrr >= best_mrr:
            best_mrr = current_mrr
            torch.save(model.state_dict(), "model/model_gin/model_checkpoint.pt")
            
    print(f"\nBest Validation MRR: {best_mrr:.4f}")
    print("Model saved to model_checkpoint.pt")


if __name__ == "__main__":
    main()