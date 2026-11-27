import torch.nn as nn 
import torch 
from config import MyConfig
class InputEmbedding(nn.Module):

    def __init__(self, config: MyConfig):
        super().__init__()
        self.max_seq_len =  config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)


    def forward(self, input_ids, position_offset: int = 0):
        _, seq_len = input_ids.shape
        if position_offset + seq_len > self.max_seq_len:
            raise ValueError
        positions = torch.arange(position_offset, position_offset + seq_len,
                                 device=input_ids.device)
        tokens = self.token_embedding(input_ids)
        pos = self.position_embedding(positions)
        return self.dropout(tokens + pos) # (B, seq len, d_model)
    