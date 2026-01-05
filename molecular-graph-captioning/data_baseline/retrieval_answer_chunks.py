import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

# IMPORTANT: Import MolGINE from your new training script
# Ensure train_gine_chunks.py is in the same folder
from train_gine_chunks import (
    MolGINE, DEVICE, TRAIN_GRAPHS, TEST_GRAPHS
)

# Path to your chunked embeddings
TRAIN_CHUNKS_PKL = "data/train_embeddings_chunked.pkl"

@torch.no_grad()
def retrieve_descriptions(model, train_data, test_data, train_emb_dict, device, output_csv):
    """
    Args:
        model: Trained GNN model
        train_data: Path to train preprocessed graphs
        test_data: Path to test preprocessed graphs
        train_emb_dict: Dictionary mapping train IDs to text embeddings (Averaged vectors)
        device: Device to run on
        output_csv: Path to save retrieved descriptions
    """
    print(f"Loading descriptions from {train_data}...")
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    # Prepare Candidate Embeddings (Training Set)
    print("Preparing candidate embeddings...")
    train_ids = list(train_emb_dict.keys())
    
    # Ensure they are tensors on the correct device
    emb_list = []
    for id_ in train_ids:
        emb = train_emb_dict[id_]
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb, dtype=torch.float)
        emb_list.append(emb)
        
    train_embs = torch.stack(emb_list).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Train/Candidate set size: {len(train_ids)}")
    
    # Process Test Graphs
    print(f"Loading test graphs from {test_data}...")
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Test set size: {len(test_ds)}")
    
    test_mol_embs = []
    test_ids_ordered = []
    
    print("Encoding test molecules...")
    for graphs in test_dl:
        graphs = graphs.to(device)
        
        # Forward pass through GINE
        mol_emb = model(graphs)
        
        # Normalize
        mol_emb = F.normalize(mol_emb, dim=-1)
        
        test_mol_embs.append(mol_emb)
        
        # Track IDs
        batch_size = graphs.num_graphs
        start_idx = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start_idx:start_idx + batch_size])
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    print(f"Encoded {test_mol_embs.size(0)} test molecules")
    
    # Compute Similarity: (Test Molecules) x (Train Text Prototypes)
    print("Calculating similarity matrix...")
    similarities = test_mol_embs @ train_embs.t()
    
    # Find nearest neighbor
    most_similar_indices = similarities.argmax(dim=-1).cpu()
    
    results = []
    print("Generating submission file...")
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = most_similar_indices[i].item()
        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })
        
        if i < 3:
            print(f"\n[DEBUG] Test ID {test_id} -> Matches Train ID {retrieved_train_id}")
            print(f"Desc: {retrieved_desc[:100]}...")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} retrieved descriptions to: {output_csv}")
    
    return results_df


def main():
    print(f"Device: {DEVICE}")
    output_csv = "test_retrieved_descriptions.csv"
    
    # 1. Load Model Checkpoint
    # Look for the best model saved by your training script
    # It might be named "model_checkpoint.pt" or "model_checkpoint_0.XXXX.pt"
    # Update this path to match your actual best file!
    model_path = "model_checkpoint_0.2633.pt" 
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        print("Please train a model first using train_gine_chunks.py")
        return
    
    if not os.path.exists(TEST_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TEST_GRAPHS}")
        return

    # 2. Load Chunk Embeddings & Compute Average
    if not os.path.exists(TRAIN_CHUNKS_PKL):
        print(f"Error: {TRAIN_CHUNKS_PKL} not found. Run generate_chunks_embeddings.py first.")
        return

    print(f"Loading chunk embeddings from {TRAIN_CHUNKS_PKL}...")
    with open(TRAIN_CHUNKS_PKL, "rb") as f:
        train_emb_dict = pickle.load(f)
        
    print("Averaging text chunks to create retrieval prototypes...")
    processed_emb_dict = {}
    
    # Get dimension from the first item to initialize model
    first_key = next(iter(train_emb_dict))
    first_val = train_emb_dict[first_key]
    
    if isinstance(first_val, (list, np.ndarray)) and len(np.shape(first_val)) > 1:
        emb_dim = len(first_val[0])
    else:
        emb_dim = len(first_val)
        
    print(f"Detected embedding dimension: {emb_dim}")

    # Process all embeddings
    for k, v in train_emb_dict.items():
        # If it's a list of chunks (matrix), take the mean across axis 0
        if isinstance(v, (list, np.ndarray)) and len(np.shape(v)) > 1:
            processed_emb_dict[k] = np.mean(v, axis=0)
        else:
            processed_emb_dict[k] = v

    # 3. Initialize & Load Model
    # Using MolGINE as used in training
    model = MolGINE(hidden=128, out_dim=emb_dim, layers=4).to(DEVICE)
    print(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 4. Run Retrieval
    retrieve_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=processed_emb_dict,
        device=DEVICE,
        output_csv=output_csv
    )

if __name__ == "__main__":
    main()