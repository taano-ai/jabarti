from dataclasses import dataclass

@dataclass
class MyConfig:
    vocab_size: int = 2000
    d_model: int = 768  # dim
    max_seq_len: int  = 1024 # context window 
    dropout: float = 0.1