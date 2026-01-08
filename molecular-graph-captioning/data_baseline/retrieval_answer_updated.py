import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

# Make sure these classes are importable from your train script
# If they are in the same directory, this should work. 
# Otherwise, copy the class definitions for MolGINE and TextAdapter here.
from train_gin import (
    MolGINE, TextAdapter, DEVICE, TRAIN_GRAPHS, TEST_GRAPHS, TRAIN_EMB_CSV
)

@torch.no_grad()
def retrieve_descriptions(graph_model, text_adapter, train_data, test_data, train_emb_dict, device, output_csv):
    """
    Args:
        graph_model: Trained MolGINE
        text_adapter: Trained TextAdapter
        train_data: Path to train preprocessed graphs
        test_data: Path to test preprocessed graphs
        train_emb_dict: Dictionary mapping train IDs to text embeddings
        device: Device to run on
        output_csv: Path to save retrieved descriptions
    """
    graph_model.eval()
    text_adapter.eval()
    
    # 1. Load Ground Truth Descriptions (for the final CSV)
    print("Loading training descriptions...")
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    # 2. Prepare Training Text Embeddings (The "Database")
    # We must project the raw 768-dim BERT embeddings into the 300-dim shared space
    print("Projecting training embeddings to latent space...")
    train_ids = list(train_emb_dict.keys())
    
    # Stack raw embeddings
    raw_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    
    # PASS THROUGH ADAPTER (Critical Step!)
    # Raw (768) -> Shared Space (300)
    train_latents = text_adapter(raw_embs)
    train_latents = F.normalize(train_latents, dim=-1)
    
    print(f"Database ready: {len(train_ids)} items in shared space.")
    
    # 3. Encode Test Molecules
    print("Encoding test molecules...")
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    test_mol_embs = []
    test_ids_ordered = []
    
    for batch in test_dl: 
        # Case A: If the loader returns a tuple (graphs, text, ids)
        # The wildcard unpacking above might have been slightly off if the collate returns a tuple directly.
        
        # Let's handle it safely:
        if isinstance(batch, (list, tuple)):
            graphs = batch[0] # The first item is usually the graph Batch object
        else:
            graphs = batch # It's already the object
            
        graphs = graphs.to(device)
        
        # Graph (Nodes/Edges) -> Shared Space (300)
        mol_emb = graph_model(graphs)
        # Note: MolGINE usually includes projection head, so this is already 300-dim
        
        test_mol_embs.append(mol_emb)
        batch_size = graphs.num_graphs
        start_idx = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start_idx:start_idx + batch_size])
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    test_mol_embs = F.normalize(test_mol_embs, dim=-1) # Normalize!
    
    print(f"Encoded {test_mol_embs.size(0)} test molecules.")
    
    # 4. Retrieval (Cosine Similarity in Shared Space)
    # [Num_Test, 300] @ [300, Num_Train] = [Num_Test, Num_Train]
    print("Calculating similarities...")
    similarities = test_mol_embs @ train_latents.t()
    
    # Find nearest neighbor for each test molecule
    most_similar_indices = similarities.argmax(dim=-1).cpu()
    
    results = []
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = most_similar_indices[i].item()
        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })
        
        if i < 3:
            print(f"\n[Preview] Test ID {test_id} -> Train ID {retrieved_train_id}")
            print(f"Desc: {retrieved_desc[:100]}...")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} retrieved descriptions to: {output_csv}")
    
    return results_df


def main():
    print(f"Device: {DEVICE}")
    output_csv = "submission.csv"
    
    # UPDATE THIS TO YOUR BEST CHECKPOINT
    model_path = "model_checkpoint_0.7362.pt" 
    
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint {model_path} not found.")
        return
    
    # 1. Initialize Models (Dimensions must match training!)
    # Assuming hidden=300 based on your latest config
    graph_model = MolGINE(hidden=300, out_dim=300).to(DEVICE)
    text_adapter = TextAdapter(input_dim=768, hidden_dim=300, output_dim=300).to(DEVICE)
    
    # 2. Load Checkpoint
    # Note: Using strict=False just in case, but keys should match if trained correctly
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Handle the dictionary saving format
    if 'graph_model' in checkpoint:
        graph_model.load_state_dict(checkpoint['graph_model'])
        text_adapter.load_state_dict(checkpoint['text_adapter'])
    else:
        # Fallback for old saves (just graph model)
        print("Warning: Old checkpoint format detected. Loading as graph model only.")
        graph_model.load_state_dict(checkpoint)

    # 3. Load Embeddings
    print("Loading embedding dictionary...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)

    # 4. Run Retrieval
    retrieve_descriptions(
        graph_model=graph_model,
        text_adapter=text_adapter,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=output_csv
    )

if __name__ == "__main__":
    main()