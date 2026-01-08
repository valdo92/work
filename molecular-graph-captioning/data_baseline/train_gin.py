import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import contrastive_loss, ArcFaceLoss, BatchHardTripletLoss
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_add_pool

from transformers import get_cosine_schedule_with_warmup
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

BATCH_SIZE = 256  # Larger batch size is better for Contrastive Loss
EPOCHS = 40       # GIN needs more epochs than the dummy baseline
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

class EdgeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Dimensions based on data_utils.e_map
        self.bond_embedding = nn.Embedding(22, hidden_dim) # bond_type
        self.stereo_embedding = nn.Embedding(6, hidden_dim) # stereo
        self.conj_embedding = nn.Embedding(2, hidden_dim)  # is_conjugated

    def forward(self, edge_attr):
        # edge_attr shape: [num_edges, 3]
        # Summing embeddings is the standard strategy (like in OGB)
        bond = self.bond_embedding(edge_attr[:, 0])
        stereo = self.stereo_embedding(edge_attr[:, 1])
        conj = self.conj_embedding(edge_attr[:, 2])
        
        return bond + stereo + conj

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

# --- 1. Trainable Text Adapter ---
class TextAdapter(nn.Module):
    """
    Feeds the frozen BERT embeddings into a trainable MLP.
    Allows the text space to align with the graph space.
    """
    def __init__(self, input_dim=768, hidden_dim=300, output_dim=300):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # Project to shared space
        )

    def forward(self, x):
        return self.mlp(x)

# --- 2. Upgraded GINE with Projection Head ---
class MolGINE(nn.Module):
    def __init__(self, hidden=300, out_dim=300, layers=5): # Increased hidden & layers
        super().__init__()
        
        # Encoders (Now hidden=300)
        self.atom_encoder = AtomEncoder(hidden)
        self.edge_encoder = EdgeEncoder(hidden)
        
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                nn.BatchNorm1d(2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            
        # PROJECTION HEAD (The SimCLR trick)
        # Instead of direct projection, we use an MLP
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim) # Project to shared space (e.g. 300)
        )

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        h = self.atom_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)
        
        for conv in self.convs:
            h_in = h
            h = conv(h, edge_index, edge_attr=edge_emb)
            h = F.relu(h)
            h = h + h_in
            
        g = global_add_pool(h, batch_idx)
        
        # Pass through Projection Head
        return self.proj_head(g)

def augment_graph(data, mask_prob=0.15, drop_prob=0.1):
    """
    1. Masking: Randomly zeros out node features.
    2. Dropping: Randomly removes edges.
    """
    data = data.clone()
    node_num = data.x.size(0)
    edge_num = data.edge_index.size(1)

    # 1. Mask Node Features
    mask = torch.rand(node_num) < mask_prob
    data.x[mask] = 0 # Assuming 0 is the padding index

    # 2. Drop Edges
    keep_edges = torch.rand(edge_num) > drop_prob
    data.edge_index = data.edge_index[:, keep_edges]
    data.edge_attr = data.edge_attr[keep_edges]

    return data

#=======================================================
# Training Loop
# =========================================================
triplet_criterion = BatchHardTripletLoss(margin=0.2).to(DEVICE)

def train_epoch(graph_model, text_adapter, loader, optimizer,scheduler, device):
    graph_model.train()
    text_adapter.train() # Don't forget to set this to train mode!
    total_loss, total = 0.0, 0
    
    for graphs, text_emb_frozen in loader:
        graphs = augment_graph(graphs).to(device)
        text_emb_frozen = text_emb_frozen.to(device)
        
        optimizer.zero_grad()
        
        # 1. Forward Pass Graph
        mol_vec = graph_model(graphs)  # Shape: [Batch, 300]
        
        # 2. Forward Pass Text (Through Adapter)
        text_vec = text_adapter(text_emb_frozen) # Shape: [Batch, 300]
        
        # 3. Contrastive Loss (Now dimensions match: 300 vs 300)
        loss = contrastive_loss(mol_vec, text_vec)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs
        
    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, graph_model, text_adapter, device):
    graph_model.eval()
    text_adapter.eval()
    
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb_frozen in dl:
        graphs = graphs.to(device)
        text_emb_frozen = text_emb_frozen.to(device)
        
        # Project BOTH to shared 300-dim space
        mol_vec = graph_model(graphs)
        text_vec = text_adapter(text_emb_frozen)
        
        all_mol.append(F.normalize(mol_vec, dim=-1))
        all_txt.append(F.normalize(text_vec, dim=-1)) # Normalize the ADAPTED text
        
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    # Similarity matrix
    sims = all_mol @ all_txt.t()
    
    # ... (Rest of ranking logic stays the same) ...
    ranks = sims.argsort(dim=-1, descending=True)
    N = all_txt.size(0)
    correct_indices = torch.arange(N, device=device).unsqueeze(1)
    pos = (ranks == correct_indices).nonzero()[:, 1] + 1
    mrr = (1.0 / pos.float()).mean().item()
    
    results = {"MRR": mrr}
    return results

def main():
    print(f"Device: {DEVICE}")

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None

    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TRAIN_GRAPHS}")
        print("Please run: python prepare_graph_data.py")
        return
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    print("preprocessed")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    total_steps = len(train_dl) * EPOCHS
    # --- 3. Updated Training Loop Snippet ---
    # Initialize BOTH models
    graph_model = MolGINE(hidden=300, out_dim=300).to(DEVICE)
    text_adapter = TextAdapter(input_dim=768, output_dim=300).to(DEVICE)

    # Optimizer handles parameters from BOTH
    optimizer = torch.optim.Adam(
        list(graph_model.parameters()) + list(text_adapter.parameters()), 
        lr=1e-3, weight_decay=1e-5
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    best_mrr = 0.0
    patience_counter =0

    for ep in range(EPOCHS):
        train_loss = train_epoch(graph_model, text_adapter, train_dl, optimizer, scheduler, DEVICE)
        
        # 2. Validation Step (Averaged Embeddings)
        # Check if the validation pickle exists
        if val_emb is not None and os.path.exists(VAL_GRAPHS):
            scores = eval_retrieval(VAL_GRAPHS, val_emb, graph_model,text_adapter, DEVICE)
        else:
            scores = {}
            # Pass the PATH string, not a dictionary
        current_mrr = scores['MRR']
        print(f"Epoch {ep+1} - Val MRR: {scores['MRR']:.4f}")
        
        print(f"Epoch {ep+1}/{EPOCHS} - Loss: {train_loss:.4f} {scores}")
        
        # Save best model
        if current_mrr >= best_mrr:
            best_mrr = current_mrr
            # SAVE BOTH MODELS
            torch.save({
                'graph_model': graph_model.state_dict(),
                'text_adapter': text_adapter.state_dict()
            }, f"model_checkpoint_{best_mrr:.4f}.pt")
        else : 
            patience_counter +=1
            if patience_counter >10:
                print("\nEarly stopping triggered! Training finished.")
                break
            
    print(f"\nBest Validation MRR: {best_mrr:.4f}")
    print("Model saved to model_checkpoint.pt")


if __name__ == "__main__":
    main()