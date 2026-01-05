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
    
    # We process all three splits
    splits = ['train', 'validation', 'test'] 

    for split in splits:
        pkl_path = f'data/{split}_graphs.pkl'
        if not os.path.exists(pkl_path):
            continue
            
        print(f"Processing {split}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)

        data_dict = {}

        for graph in tqdm(graphs):
            # 1. Chunking Strategy
            raw_desc = graph.description
            # Split by period, keep chunks with actual content (>15 chars)
            chunks = [s.strip() for s in raw_desc.split('.') if len(s.strip()) > 15]
            if len(chunks) == 0: chunks = [raw_desc]
            
            # 2. Encode All Chunks
            with torch.no_grad():
                inputs = tokenizer(chunks, return_tensors='pt', padding=True, 
                                 truncation=True, max_length=MAX_LEN).to('cuda')
                outputs = model(**inputs)
                # Shape: [Num_Chunks, 768]
                chunk_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 3. SAVE STRATEGY
            if split == 'train':
                # TRAIN: Save the LIST of chunks (so we can pick randomly later)
                data_dict[graph.id] = chunk_embs
            else:
                # VAL/TEST: Save the AVERAGE of chunks (Your idea!)
                # Shape: [768]
                avg_emb = np.mean(chunk_embs, axis=0)
                data_dict[graph.id] = avg_emb

        # Save as Pickle
        output_path = f'data/{split}_embeddings_chunked.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
        print(f"Saved {split} to {output_path}")

if __name__ == "__main__":
    generate_embeddings()