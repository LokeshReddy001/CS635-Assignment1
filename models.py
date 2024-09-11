# models.py

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertModel

class BertClsFFN(nn.Module):
    """
        A small feed forward network on top of CLS embedding, to get a score
    """
    def __init__(self):
        super(BertClsFFN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.ffn = nn.Sequential(
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Linear(8, 1),

        )
        self.freeze_bert()

    
    def freeze_bert(self):
        self.bert.embeddings.requires_grad_(False)
        for param in self.bert.encoder.layer[:11].parameters():
            param.requires_grad = False

    def forward(self, input_tokens):
        sentence_embed = self.bert(**input_tokens).pooler_output
        scores = self.ffn(sentence_embed).reshape(-1)
        return scores

class BertLogitScorer(nn.Module):
    """
        If the input format is [CLS] `sent1` [SEP] `sent2` [SEP], we sum the log_probs of tokens of `sent2` to get a representation of a score
    """
    def __init__(self):
        super(BertLogitScorer, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.freeze_bert()

    def freeze_bert(self):
        self.bert.bert.embeddings.requires_grad_(False)
        for param in self.bert.bert.encoder.layer[:10].parameters():
            param.requires_grad = False

    def forward(self, input_tokens):
        input_ids = input_tokens['input_ids']
        batch_size, seq_length = input_ids.shape
        logits = self.bert(**input_tokens).logits
        log_probs = F.log_softmax(logits, dim=-1)

        sums = torch.zeros(batch_size, device=logits.device)

        for i in range(batch_size):
            sep_indices = (input_ids[i] == 102).nonzero(as_tuple=True)[0]
            idx1, idx2 = sep_indices[0].item(), sep_indices[1].item()
            token_ids_in_range = input_ids[i, idx1 + 1:idx2]
            log_probs_in_range = log_probs[i, idx1 + 1:idx2]
            gathered_log_probs = torch.gather(log_probs_in_range, dim=1, index=token_ids_in_range.unsqueeze(-1)).squeeze(-1)
            
            sums[i] = torch.sum(gathered_log_probs)

        return sums
