from dataclasses import dataclass

# max_seq_length he said context window, 
# 2) he put it in positional embedding  
# 3) position offeset in forward why ? 4) what is dropout ? 
@dataclass
class MyConfig:
    vocab_size: int = 2000
    d_model: int = 768  # dim
    max_seq_len: int  = 1024
    dropout: float = 0.1