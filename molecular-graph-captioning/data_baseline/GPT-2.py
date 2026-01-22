import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch_geometric.nn import GINEConv, global_add_pool

class GraphGPT2(nn.Module):
    def __init__(self, gine_hidden_dim=300, gpt2_model='gpt2'):
        super(GraphGPT2, self).__init__()
        
        #GINE Encoder cf retrieval
        self.emb_atom = nn.Embedding(120, gine_hidden_dim)
        self.emb_bond = nn.Embedding(25, gine_hidden_dim)
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(gine_hidden_dim, gine_hidden_dim*2), 
                    nn.ReLU(), 
                    nn.Linear(gine_hidden_dim*2, gine_hidden_dim)
                ), train_eps=True
            ) for _ in range(3)
        ])
        
        #GPT-2 Decoder
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        
        #Freezing GPT-2 weights (just to save memory)
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        #Projection Layer
        self.gpt_dim = self.gpt2.config.n_embd
        self.project_graph = nn.Linear(gine_hidden_dim, self.gpt_dim)

    def forward(self, data, input_ids, attention_mask):
        #Encode Graph
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.emb_atom(x[:,0].long().clamp(max=119))
        e = self.emb_bond(edge_attr[:,0].long().clamp(max=24))
        
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index, edge_attr=e))
        
        #Global Pooling
        z_g = global_add_pool(h, data.batch) # Shape: [Batch, 300]
        
        #Project to GPT-2 Space
        graph_embedding = self.project_graph(z_g).unsqueeze(1) # Shape: [Batch, 1, 768]
        
        #Text Embeddings
        wte = self.gpt2.transformer.wte
        text_embeddings = wte(input_ids) 
        
        #Concatenation
        inputs_embeds = torch.cat((graph_embedding, text_embeddings), dim=1)
        
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=None)
        
        return outputs.logits

    def generate_caption(self, data, max_length=50):
        with torch.no_grad():
            #Graph Vector
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            h = self.emb_atom(x[:,0].long())
            e = self.emb_bond(edge_attr[:,0].long())
            for conv in self.convs: h = torch.relu(conv(h, edge_index, edge_attr=e))
            z_g = global_add_pool(h, data.batch)
            
            current_embed = self.project_graph(z_g).unsqueeze(1)
            
            generated_ids = []
            for _ in range(max_length):
                outputs = self.gpt2(inputs_embeds=current_embed)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                generated_ids.append(next_token.item())
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                next_embed = self.gpt2.transformer.wte(next_token).unsqueeze(0)
                current_embed = torch.cat((current_embed, next_embed), dim=1)
                
            return self.tokenizer.decode(generated_ids)
