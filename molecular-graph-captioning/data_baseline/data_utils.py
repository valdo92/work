"""
Data loading and processing utilities for molecule-text retrieval.
Includes dataset classes and data loading functions.
"""
from typing import Dict
import pickle

import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# =========================================================
# Feature maps for atom and bond attributes
# =========================================================
from typing import Dict, List, Any

x_map: Dict[str, List[Any]] = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
        'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': [
        'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


# =========================================================
# Load precomputed text embeddings
# =========================================================
def load_id2emb(csv_path: str) -> Dict[str, torch.Tensor]:
    """
    Load precomputed text embeddings from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: ID, embedding
                  where embedding is comma-separated floats
        
    Returns:
        Dictionary mapping ID (str) to embedding tensor
    """
    df = pd.read_csv(csv_path)
    id2emb = {}
    for _, row in df.iterrows():
        id_ = str(row["ID"])
        emb_str = row["embedding"]
        emb_vals = [float(x) for x in str(emb_str).split(',')]
        id2emb[id_] = torch.tensor(emb_vals, dtype=torch.float32)
    return id2emb


# =========================================================
# Load descriptions from preprocessed graphs
# =========================================================
def load_descriptions_from_graphs(graph_path: str) -> Dict[str, str]:
    """
    Load ID to description mapping from preprocessed graph file.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        
    Returns:
        Dictionary mapping ID (str) to description (str)
    """
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    
    id2desc = {}
    for graph in graphs:
        id2desc[graph.id] = graph.description
    
    return id2desc


# =========================================================
# Dataset that loads preprocessed graphs and text embeddings
# =========================================================
class PreprocessedGraphDataset(Dataset):
    def __init__(self, graph_path: str, emb_file_path=None, train_mode: bool = False, emb_dict=None):
        """
        Args:
            graph_path: Path to graph pickle file.
            emb_file_path: Path to embedding PKL/CSV OR a Dictionary (legacy support).
            train_mode: If True, randomly picks one embedding chunk per epoch.
            emb_dict: Explicit dictionary argument (optional).
        """
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        self.ids = [g.id for g in self.graphs]
        self.train_mode = train_mode
        self.emb_dict = emb_dict

        # --- SMART DETECTION LOGIC ---
        # 1. Handle Legacy Call: PreprocessedGraphDataset(graphs, embedding_dictionary)
        if isinstance(emb_file_path, dict):
            self.emb_dict = emb_file_path
            emb_file_path = None # It's not a path, it's data

        # 2. Handle New Call: PreprocessedGraphDataset(graphs, "path/to/file.pkl")
        if emb_file_path and isinstance(emb_file_path, str):
            print(f"Loading embeddings from {emb_file_path}...")
            if emb_file_path.endswith('.pkl'):
                with open(emb_file_path, 'rb') as f:
                    self.emb_dict = pickle.load(f)
            else:
                # Assume CSV
                self.emb_dict = load_id2emb(emb_file_path)
        
        if self.emb_dict:
            print(f"Loaded embeddings for {len(self.emb_dict)} molecules")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        
        if self.emb_dict is not None:
            id_ = graph.id
            if id_ in self.emb_dict:
                stored_data = self.emb_dict[id_]
                
                # Check if it is a list of chunks (3D array from pickle)
                if isinstance(stored_data, (list, np.ndarray)) and len(np.shape(stored_data)) > 1:
                    if self.train_mode:
                        # TRAINING CHUNKS: Pick one randomly
                        num_chunks = len(stored_data)
                        rand_idx = np.random.randint(0, num_chunks)
                        text_emb = torch.tensor(stored_data[rand_idx], dtype=torch.float32)
                    else:
                        # VALIDATION CHUNKS: Average them
                        mean_emb = np.mean(stored_data, axis=0)
                        text_emb = torch.tensor(mean_emb, dtype=torch.float32)
                else:
                    # STANDARD/LEGACY: Single vector (from CSV or averaged pickle)
                    if not torch.is_tensor(stored_data):
                        text_emb = torch.tensor(stored_data, dtype=torch.float32)
                    else:
                        text_emb = stored_data
            else:
                # Fallback for missing IDs
                text_emb = torch.zeros(768) 

            return graph, text_emb
        else:
            return graph

def collate_fn(batch):
    """
    Collate function for DataLoader to batch graphs with optional text embeddings.
    
    Args:
        batch: List of graph Data objects or (graph, text_embedding) tuples
        
    Returns:
        Batched graph or (batched_graph, stacked_text_embeddings)
    """
    if isinstance(batch[0], tuple):
        graphs, text_embs = zip(*batch)
        batch_graph = Batch.from_data_list(list(graphs))
        text_embs = torch.stack(text_embs, dim=0)
        return batch_graph, text_embs
    else:
        return Batch.from_data_list(batch)

