import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from rdkit import Chem
from torch.utils.data import Dataset
import pandas as pd
import pickle

class MoleculesDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_source_length=128, max_target_length=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def graph_to_smiles(self, data):
        try:
            mol = Chem.RWMol()
            node_to_idx = {}
            x = data.x.cpu().numpy()
            
            for i in range(len(x)):
                atomic_num = int(x[i, 0])
                if atomic_num == 0: atomic_num = 6 
                idx = mol.AddAtom(Chem.Atom(atomic_num))
                node_to_idx[i] = idx
                
            edge_index = data.edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src > dst: continue
                mol.AddBond(node_to_idx[src], node_to_idx[dst], Chem.BondType.SINGLE)
                
            return Chem.MolToSmiles(mol)
        except:
            return ""

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        smiles = self.graph_to_smiles(data)
        text = data.description
        
        input_text = "describe molecule: " + smiles
        
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_source_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            text, 
            max_length=self.max_target_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }

def train_t5_model(train_data, output_dir="./results"):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    dataset = MoleculesDataset(train_data, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5, #par manque de temps
        per_device_train_batch_size=16,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
