# %%
# import libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
# Math
import math
import time

# %%
# Object-Oriented Programming of Attention Model


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 1000, embed_size: int = 512) -> torch.Tensor:
        """
        Class to create input embedding for the input tokens
        Args:
            vocab_size (int): size of the vocabulary of the dictionary. Defaults to 1000.
            embed_size (int):dimension of the embeddings. Defaults to 512.
        """
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size)

    def forward(self, input_token: torch.Tensor) -> torch.Tensor:
        input_embed = self.embedding(
            input_token) * math.sqrt(self.embedding_dim)
        return input_embed


class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len: int, embedding_dim: int = 512, drop: float = 0.2) -> torch.Tensor:
        """
        Class to create positional encoding for the input tokens
        Args:
            max_seq_len (int): Maximum sequence length 
            embedding_dim (int): dimension of the embeddings. The default is 512.
            drop(float): Dropout rate to apply to positional encoding. The default is 0.2.
        """
        super(PositionEncoding, self).__init__()

        # create a tensor of shape (max_len, 1) containing values from 0 to max_len-1
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)

        # Compute the division term for the positional encoding formula
        div_term = torch.exp(torch.arange(
            0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))

        # Compute the positional encoding
        position_encoding = torch.zeros(max_seq_len, embedding_dim)

        # compute the positional encoding for even indices
        position_encoding[:, 0::2] = torch.sin(positions * div_term)

        # compute the positional encoding for odd indices
        position_encoding[:, 1::2] = torch.cos(positions * div_term)

        # Register buffer so that it's not considered a model parameter but moves with the model
        self.register_buffer('position_encoding',
                             position_encoding.unsqueeze(0))
        
        # Dropout layer
        self.dropout = nn.Dropout(p=drop)

    def forward(self, input_embed_token: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input_embed_token (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: Positional encoded input of the same shape.
        """
        x = input_embed_token + \
            self.position_encoding[:, :input_embed_token.size(
                1), :]
        return self.dropout(x)


class Add_Norm(nn.Module):
    def __init__(self, embed_size: int , eps: float = 1e-6) -> None:
        """
        Class to add and normalize the input tensor
        Args:
            embed_size (int): dimension of the embeddings
            eps (float): Small value to prevent division by zero in normalization. The default is 1e-6.
        """
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_size, eps=eps)
    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Add & Norm.

        Args:
            x (torch.Tensor): Input tensor (residual connection).
            sublayer_output (torch.Tensor): Output of the sublayer (e.g., attention or feedforward).

        Returns:
            torch.Tensor: Normalized tensor after applying residual connection.
        """
        x = self.norm(x + sublayer_output)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, embed_size: int = 512, ff_hidden_size: int = 2048, drop: float = 0.2) -> None:
        """
        Class to create the feedforward network
        Args:
            embed_size (int): dimension of the embeddings. The default is 512.
            ff_hidden_size (int): Hidden size of the feedforward network. The default is 2048.
            drop (float): Dropout rate. The default is 0.2.
        """
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.GELU(), # Switched from ReLU to GELU
            nn.Linear(ff_hidden_size, embed_size),
            nn.Dropout(drop), # Dropout is applied after the final linear layer
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feedforward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        return self.ff(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int = 512, heads: int = 8, drop: float = 0.2) -> None:
        """
        Class to create the multi-head attention mechanism
        Args:
            embed_size (int): dimension of the embeddings. The default is 512.
            heads (int): Number of attention heads. The default is 8.
            drop (float): Dropout rate. The default is 0.2.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Check if embed_size is divisible by heads
        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        # Query, Key, Value and Output weight matrices
        self.q_linear = nn.Linear(self.embed_size, self.embed_size)
        self.k_linear = nn.Linear(self.embed_size, self.embed_size)
        self.v_linear = nn.Linear(self.embed_size, self.embed_size)
        self.o_linear = nn.Linear(self.embed_size, self.embed_size)

        # Dropout layer
        self.dropout = nn.Dropout(drop)

    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the multi-head attention mechanism.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor. The default is None.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size = query.shape[0]

        # Linear transformation for query, key and value
        Q = self.q_linear(query) # Q` = Q * W_q   (Batch, seq_len, embed_size)
        K = self.k_linear(key) # K` = K * W_k   (Batch, seq_len, embed_size)
        V = self.v_linear(value) # V` = V * W_v   (Batch, seq_len, embed_size)

        # Split the embedding into self.heads , self.head_dim 
        # and then concatenate them to get the desired number of heads
        Q = Q.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3) # (Batch, heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3) # (Batch, heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3) # (Batch, heads, seq_len, head_dim)

        # Compute the scaled dot-product attention
        # Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k)) * V
        # where d_k is the dimension of the key
        d_k = Q.shape[-1]

        # Compute the attention score matrix (Q*K^T/sqrt(d_k))
        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(d_k) # (Batch, heads, seq_len, seq_len)

        # apply mask if provided
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-1e20'))
        
        # Apply softmax to get the attention weights
        attention_score = torch.softmax(attention_score, dim=-1) # (Batch, heads, seq_len, seq_len)

        # Apply dropout 
        if self.dropout is not None:
            attention_score = self.dropout(attention_score)

        # Compute the output of the attention mechanism 
        attention  = torch.matmul(attention_score, V) # (Batch, heads, seq_len, head_dim)

        # Concat the heads to get the original embedding size 
        attention = attention.permute(0, 2, 1, 3).contiguous() # (Batch, seq_len, heads, head_dim)

        # Reshape the attention tensor
        attention = attention.view(batch_size, -1, self.embed_size) # (Batch, seq_len, embed_size)

        # Apply the output linear layer
        output = self.o_linear(attention) # (Batch, seq_len, embed_size)

        return output




