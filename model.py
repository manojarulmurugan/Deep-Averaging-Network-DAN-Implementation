import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        print(f'Loading model from {path}')
        ckpt = torch.load(path,weights_only=False)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])
  
def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size)
    """
    emb = np.random.uniform(-0.05, 0.05, (len(vocab), emb_size)).astype(np.float32)
    found = 0
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            if word in vocab:
                #vec = np.array([float(x) for x in parts[1:]])
                vec = parts[-emb_size:]
                if len(vec) != emb_size:
                    continue
                '''if vec.shape[0] == emb_size:
                    emb[vocab[word]] = vec
                    found += 1'''
                try:
                    emb[vocab[word]] = np.asarray(vec, dtype=np.float32)
                    found += 1
                except Exception:
                    pass
    print(f"Loaded embeddings for {found}/{len(vocab)} words.")
    print(f"from {emb_file}")
    return emb

# Attention Head:
'''class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, emb, mask):
        scores = self.proj(emb).squeeze(-1)         
        scores = scores.masked_fill(mask == 0, -1e9)  
        attn = torch.softmax(scores, dim=1)  
        pooled = (emb * attn.unsqueeze(-1)).sum(dim=1)
        return pooled'''


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        #self.freeze_emb = getattr(self.args, "freeze_emb", False) #Freezing pretrained embeddings
        self.define_model_parameters()
        self.init_model_parameters()

        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)
        self.emb_dropout = nn.Dropout(self.args.emb_drop)

        #Attention Mechanism
        self.num_heads = getattr(self.args, 'num_heads', 2)
        self.attn_heads = nn.ModuleList([
            nn.Linear(self.args.emb_size, 1, bias=False) for _ in range(self.num_heads)
        ])
        self.fc1 = nn.Linear(self.args.emb_size*(2 + self.num_heads), self.args.hid_size)
        self.fc2 = nn.Linear(self.args.hid_size, self.tag_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.args.hid_drop)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        v = 0.08
        nn.init.uniform_(self.embedding.weight, -v, v)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb_matrix = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        emb = self.embedding(x)               
        emb = self.emb_dropout(emb)
        
        attn_pooled = []
        for head in self.attn_heads:
            attn_scores = head(emb).squeeze(-1)      # [batch, seq_len]
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
            pooled = torch.sum(emb * attn_weights, dim=1)
            attn_pooled.append(pooled)

        attn_pooled = torch.cat(attn_pooled, dim=1)  # [batch, emb_size * num_heads]

        # Hybrid pooling: attention + max + sum
        max_pool, _ = torch.max(emb, dim=1)
        sum_pool = torch.sum(emb, dim=1)
        pooled_emb = torch.cat([attn_pooled, max_pool, sum_pool], dim=1)  # [batch, emb_size*(num_heads +2)]

        '''#Pooling (masked):
        pm = self.args.pooling_method
        if pm == "avg":
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / lengths
        elif pm == "sum":
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1)
        elif pm == "max":
            neg_inf = torch.finfo(emb.dtype).min
            emb_masked = emb.masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
            pooled, _ = emb_masked.max(dim=1)
        else:  # "attn"
            pooled = self.attn_pool(emb, mask)'''

        h = self.act(self.fc1(pooled_emb))
        h = self.dropout(h)
        scores = self.fc2(h)
        return scores
