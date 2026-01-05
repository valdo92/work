import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50):
        """
        ArcFace Loss for Retrieval
        Args:
            s (float): Scale factor (inverse temperature). Typical values: 30-64.
            m (float): Angular margin. Typical values: 0.3-0.5.
        """
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Thresholds to avoid numerical instability
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, mol_emb, text_emb):
        """
        Input:
            mol_emb: [batch_size, emb_dim] (Graph Embeddings)
            text_emb: [batch_size, emb_dim] (Text Embeddings)
        """
        # 1. Normalize both embeddings to place them on the hypersphere
        mol_emb = F.normalize(mol_emb, dim=1)
        text_emb = F.normalize(text_emb, dim=1)

        # 2. Compute Cosine Similarity (Cosine of angle theta)
        # shape: [batch_size, batch_size]
        cosine = torch.matmul(mol_emb, text_emb.t())

        # 3. Create Ground Truth Labels (Diagonal elements are positive pairs)
        # labels: [0, 1, 2, ..., batch_size-1]
        labels = torch.arange(cosine.size(0)).to(cosine.device)

        # 4. Add Margin to the Positive Pairs (Diagonal)
        # We need to apply the margin ONLY to the ground truth (diagonal) elements.
        
        # Get the diagonal (positive pairs)
        # Note: We must clamp cosine to [-1, 1] to avoid nan in acos
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Handle numerical stability for angles > pi - m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 5. Create the logits
        # We want to replace the diagonal of 'cosine' with 'phi'
        # Create a one-hot mask for the diagonal
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Final logits: Scale * (Cosine with margin for positives + Original cosine for negatives)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # 6. Standard Cross Entropy on the modified logits
        return F.cross_entropy(output, labels)

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, mol_emb, text_emb):
        """
        Args:
            mol_emb: [batch_size, dim]
            text_emb: [batch_size, dim]
        """
        # 1. Normalize embeddings (Crucial for cosine distance)
        mol_emb = F.normalize(mol_emb, dim=1)
        text_emb = F.normalize(text_emb, dim=1)

        # 2. Compute Similarity Matrix (Cosine Similarity)
        # scores[i, j] = similarity between Mol i and Text j
        scores = torch.matmul(mol_emb, text_emb.t())
        
        # 3. Get Positive Scores (Diagonal)
        # The score of Mol i with its correct Text i
        pos_scores = torch.diag(scores)
        
        # 4. Get Hardest Negative Scores
        # We want the highest similarity among the WRONG answers.
        # We mask the diagonal (correct answers) so we don't pick them.
        batch_size = mol_emb.size(0)
        mask = torch.eye(batch_size, device=mol_emb.device).bool()
        
        # Set diagonal to -infinity so it's never chosen as the "highest" negative
        scores_masked = scores.clone()
        scores_masked.masked_fill_(mask, -float('inf'))
        
        # Max score across each row = The hardest negative for that molecule
        hardest_neg_scores, _ = scores_masked.max(dim=1)
        
        # 5. Compute Triplet Loss
        # Loss = Max(0, Margin + Negative_Score - Positive_Score)
        # We want Positive_Score > Negative_Score + Margin
        losses = F.relu(self.margin + hardest_neg_scores - pos_scores)
        
        return losses.mean()