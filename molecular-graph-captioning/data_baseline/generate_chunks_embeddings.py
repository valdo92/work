import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

MODEL_NAME = 'allenai/scibert_scivocab_uncased'
MAX_LEN = 64

def generate_embeddings():
    print("Loading SciBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).cuda().eval()
    
    # === CHANGE HERE: Removed 'test' from this list ===
    # The test set graphs do not have descriptions, so we cannot embed them.
    splits = ['train', 'validation'] 

    for split in splits:
        pkl_path = f'data/{split}_graphs.pkl'
        if not os.path.exists(pkl_path):
            print(f"Skipping {split} (file not found)")
            continue
            
        print(f"Processing {split}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)

        data_dict = {}

        for graph in tqdm(graphs):
            # Safety check: ensure description exists
            if not hasattr(graph, 'description') or graph.description is None:
                continue

            # 1. Chunking Strategy
            raw_desc = graph.description
            chunks = [s.strip() for s in raw_desc.split('.') if len(s.strip()) > 15]
            if len(chunks) == 0: chunks = [raw_desc]
            
            # 2. Encode All Chunks
            with torch.no_grad():
                inputs = tokenizer(chunks, return_tensors='pt', padding=True, 
                                 truncation=True, max_length=MAX_LEN).to('cuda')
                outputs = model(**inputs)
                chunk_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 3. SAVE STRATEGY
            if split == 'train':
                # TRAIN: Save List of chunks (for random sampling)
                data_dict[graph.id] = chunk_embs
            else:
                # VALIDATION: Save Average vector (for stable ranking)
                avg_emb = np.mean(chunk_embs, axis=0)
                data_dict[graph.id] = avg_emb

        # Save as Pickle
        output_path = f'data/{split}_embeddings_chunked.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
        print(f"Saved {split} to {output_path}")

if __name__ == "__main__":
    generate_embeddings()