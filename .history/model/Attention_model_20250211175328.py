# %%
# import libraries
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader, random_split # type: ignore
from torch.utils.tensorboard import SummaryWriter # type: ignore
# Math
import math
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# Object-Oriented Programming of Attention Model

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 50000, embed_size: int = 512) -> torch.Tensor:
        """
        Class to create input embedding for the input tokens
        Args:
            vocab_size (int): size of the vocabulary of the dictionary. Defaults to 1000.
            embed_size (int):dimension of the embeddings. Defaults to 512.
        """
        super(InputEmbedding, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_size)

    def forward(self, input_token: torch.Tensor) -> torch.Tensor:
        """
        Converts token IDs of a sentence into embedding vectors

        Args:
            input_token (torch.Tensor): the size of the input token is (1,max_seq_len,)

        Returns:
            torch.Tensor: Embedded tokens. the size of the output is (max_seq_len, embed_size)
        """
        # print('input of InputEmbedding class')
        # print(f'input_token.shape: {input_token.shape}')
        input_embed = self.embedding(
            input_token) * math.sqrt(self.embed_size)
        # print('output of InputEmbedding class')
        # print(f'input_embed.shape: {input_embed.shape}')
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
            torch.Tensor: Positional encoded input of the same shape. The output shape is (batch_size, seq_len, embedding_dim)
        """
        assert input_embed_token.size(1) <= self.position_encoding.size(1), \
        f"Sequence length {input_embed_token.size(1)} exceeds position encoding length {self.position_encoding.size(1)}"
        # print('*'*100)
        # print('shape of parameters in PositionEncoding')
        # print(f'input_embed_token.shape:{input_embed_token.shape}')
        # print(f'self.position_encoding[:, :input_embed_token.size(1), :].shape:{self.position_encoding[:, :input_embed_token.size(1), :].shape}')
        x = input_embed_token + \
            (self.position_encoding[:, :input_embed_token.size(1), :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        """
        Class to create the layer normalization layer
        Args:
            eps (float): Epsilon value. The default is 1e-6.
        """
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable parameter
        self.beta = nn.Parameter(torch.zeros(1))  # Learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the layer normalization layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor. IT keeps the shape of the input data
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


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
            nn.GELU(),  # Switched from ReLU to GELU
            nn.Linear(ff_hidden_size, embed_size),
            # Dropout is applied after the final linear layer
            nn.Dropout(drop),
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
    def __init__(self, embed_size: int = 512, num_heads: int = 8, drop: float = 0.2) -> None:
        """
        Class to create the multi-head attention mechanism
        Args:
            embed_size (int): dimension of the embeddings. The default is 512.
            num_heads (int): Number of attention heads. The default is 8.
            drop (float): Dropout rate. The default is 0.2.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = num_heads
        self.head_dim = embed_size // num_heads

        # Check if embed_size is divisible by heads
        assert self.head_dim * num_heads == embed_size, "Embedding size needs to be divisible by heads"

        # Query, Key, Value and Output weight matrices
        self.q_linear = nn.Linear(self.embed_size, self.embed_size)
        self.k_linear = nn.Linear(self.embed_size, self.embed_size)
        self.v_linear = nn.Linear(self.embed_size, self.embed_size)
        self.o_linear = nn.Linear(self.embed_size, self.embed_size)

        # Dropout layer
        self.dropout = nn.Dropout(drop)

        # initialize attention score to None
        self.attention_weights = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the multi-head attention mechanism.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor to mask padded tokens in encoder and future tokens as well as padded tokens in decoder. The default is None.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size = query.shape[0]
        # Ensure input tensors are of type float
        query = query.float()
        key = key.float()
        value = value.float()
        # Linear transformation for query, key and value
        Q = self.q_linear(query)  # Q` = Q * W_q   (Batch, seq_len, embed_size)
        K = self.k_linear(key)  # K` = K * W_k   (Batch, seq_len, embed_size)
        V = self.v_linear(value)  # V` = V * W_v   (Batch, seq_len, embed_size)

        # Split the embedding into self.heads , self.head_dim
        # and then concatenate them to get the desired number of heads
        # (Batch, heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
                   
        # (Batch, heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
                   
        # (Batch, heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
                   

        # Compute the scaled dot-product attention
        # Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k)) * V
        # where d_k is the dimension of the key
        d_k = Q.shape[-1]

        # Compute the attention score matrix (Q*K^T/sqrt(d_k))
        # (Batch, heads, seq_len, seq_len)
        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(d_k)
            

        # apply mask if provided
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e20)
                
        # Apply softmax to get the attention weights
        # (Batch, heads, seq_len, seq_len)
        attention_score = torch.softmax(attention_score, dim=-1)

        # Apply dropout
        if self.dropout is not None:
            attention_score = self.dropout(attention_score)

        self.attention_weights = attention_score

        # Compute the output of the attention mechanism
        # (Batch, heads, seq_len, head_dim)
        attention = torch.matmul(attention_score, V)

        # Concat the heads to get the original embedding size
        # (Batch, seq_len, heads, head_dim)
        attention = attention.permute(0, 2, 1, 3).contiguous()

        # Reshape the attention tensor
        # (Batch, seq_len, embed_size)
        attention = attention.view(batch_size, -1, self.embed_size)

        # Apply the output linear layer
        output = self.o_linear(attention)  # (Batch, seq_len, embed_size)
        return output

    def get_attention_weights(self) -> torch.Tensor:
        """Method to access the stored attention weights (scores)."""
        return self.attention_weights

# Residual connection layer


class ResidualConnection(nn.Module):
    def __init__(self, drop: float = 0.2) -> None:
        """
        Class to create the residual connection layer
        Args:
            embed_size (int): dimension of the embeddings
            drop (float): Dropout rate. The default is 0.2.
        """
        super(ResidualConnection, self).__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Forward pass for the residual connection layer.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (nn.Module): MultiHeadAttention or FeedForward layer

        Returns:
            torch.Tensor: Output tensor.
        """
        # Apply normalization and pass the result through the sublayer
        output = sublayer(self.norm(x))
        # Apply dropout to the output and add it to the input tensor
        return x + self.dropout(output)


# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, embed_size: int = 512, heads: int = 8, ff_hidden_size: int = 2048, drop: float = 0.2) -> None:
        """
        Class to create the encoder block
        Args:
            embed_size (int): dimension of the embeddings. The default is 512.
            heads (int): Number of attention heads. The default is 8.
            ff_hidden_size (int): Hidden size of the feedforward network. The default is 2048.
            drop (float): Dropout rate. The default is 0.2.
        """
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads, drop)
        self.residual_connection_1 = ResidualConnection(drop)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, drop)
        self.residual_connection_2 = ResidualConnection(drop)

    def forward(self, x: torch.Tensor, encoder_mask: torch.Tensor, return_attention_scores: bool = False) -> torch.Tensor:
        """
        Forward pass for the encoder block

        Args:
            x (torch.Tensor): Input tensor.
            encoder_mask (torch.Tensor): Mask teh padding tokens from calculations.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Pass attention output through the residual connection
        x = self.residual_connection_1(
            x, lambda x: self.attention(x, x, x, encoder_mask))

        # Pass through feed-forward layer and second residual connection
        x = self.residual_connection_2(x, self.feed_forward)
        if return_attention_scores:
            return x, self.attention.get_attention_weights()
        else:
            return x

# Encoder class


class Encoder(nn.Module):
    def __init__(self, layers: int, embed_size: int = 512, heads: int = 8, ff_hidden_size: int = 2048, drop: float = 0.2) -> None:
        """
        Class to create the encoder
        Args:
            layers (int): Number of encoder layers.
            embed_size (int): dimension of the embeddings. The default is 512.
            heads (int): Number of attention heads. The default is 8.
            ff_hidden_size (int): Hidden size of the feedforward network. The default is 2048.
            drop (float): Dropout rate. The default is 0.2.
        """
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(embed_size, heads, ff_hidden_size, drop) for _ in range(layers)])
        self.norm = LayerNormalization()

    def forward(self, x, encoder_mask):
        """
        Forward pass for the encoder

        Args:
            x (torch.Tensor): embedded tokens.
            mask (torch.Tensor): encoder mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for encoder in self.encoder_layers:
            x = encoder(x, encoder_mask)
        return self.norm(x)

# Decoder Block


class DecoderBlock(nn.Module):
    def __init__(self,  embed_size: int = 512, heads: int = 8, ff_hidden_size: int = 2048, drop: float = 0.2
                 ) -> None:
        """
        Class to create the decoder block
        Args:
            embed_size (int): dimension of the embeddings. The default is 512.
            heads (int): Number of attention heads. The default is 8.
            ff_hidden_size (int): Hidden size of the feedforward network. The default is 2048.
            drop (float): Dropout rate. The default is 0.2.
        """
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(
            embed_size, heads, drop)  # self_attention
        self.cross_attention = MultiHeadAttention(
            embed_size, heads, drop)  # cross_attention
        self.feed_forward = FeedForward(
            embed_size, ff_hidden_size, drop)  # feed_forward
        self.residual_connection_1 = ResidualConnection(drop)
        self.residual_connection_2 = ResidualConnection(drop)
        self.residual_connection_3 = ResidualConnection(drop)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        """
        Forward pass for the decoder block

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Output from the encoder.
            encoder_mask (torch.Tensor): Source mask tensor.
            decoder_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Pass through the self-attention mechanism
        x = self.residual_connection_1(
            x, lambda x: self.self_attention(x, x, x, decoder_mask))

        x = self.residual_connection_2(x, lambda x: self.cross_attention(
            x, encoder_output, encoder_output, encoder_mask))

        # Pass through the feed-forward network
        x = self.residual_connection_3(x, self.feed_forward)

        return x

# Decoder class


class Decoder(nn.Module):
    def __init__(self, layers: int, embed_size: int, heads: int, ff_hidden_size: int, drop: float) -> None:
        """
        Class to create the decoder
        Args:
            layers (int): Number of decoder layers.
            embed_size (int): dimension of the embeddings. The default is 512.
            heads (int): Number of attention heads. The default is 8.
            ff_hidden_size (int): Hidden size of the feedforward network. The default is 2048.
            drop (float): Dropout rate. The default is 0.2.
        """
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, ff_hidden_size, drop) for _ in range(layers)])
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        """
        Forward pass for the decoder

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Output from the encoder.(Batch, Seq_len, embed_dim)
            encoder_mask (torch.Tensor): Source mask tensor. (Batch, 1, Seq_len)
            decoder_mask (torch.Tensor): Target mask tensor. (Batch, 1, Seq_len, Seq_len)

        Returns:
            torch.Tensor: Output tensor.
        """
        for decoder in self.decoder_layers:
            x = decoder(x, encoder_output, encoder_mask, decoder_mask)
        return self.norm(x)

# Projection Layer


class ProjectionLayer(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int) -> None:
        """
        Class to create the classification head
        Args:
            embed_size (int): dimension of the embeddings.
            vocab_size (int): size of the vocabulary of the dictionary.
        """
        super(ProjectionLayer, self).__init__()
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        """
        Forward pass for the classification head

        Args:
            x (torch.Tensor): Input tensor. (Batch, seq_len, embed_size)

        Returns:
            torch.Tensor: Output tensor. (Batch, seq_len, vocab_size)
        """
        # (Batch, seq_len, embed_size) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.fc(x), dim=-1)

# Transformer Model


class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder, decoder: Decoder,
                 encoder_embed: InputEmbedding, decoder_embed: InputEmbedding,
                 encoder_position: PositionEncoding, decoder_position: PositionEncoding,
                 projection_layer: ProjectionLayer) -> None:
        """
        create a complete transformer model

        Args:
            encoder (Encoder): _description_
            decoder (Decoder): _description_
            encoder_embed (InputEmbedding): _description_
            decoder_embed (InputEmbedding): _description_
            encoder_position (PositionEncoding): _description_
            decoder_position (PositionEncoding): _description_
            projection (ProjectionLayer): _description_
            source_pad_idx (int): _description_
            target_pad_idx (int): _description_
        """
        super().__init__()  # This must be called first!
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_embed = encoder_embed
        self.decoder_embed = decoder_embed
        self.encoder_position = encoder_position
        self.decoder_position = decoder_position
        self.projection_layer = projection_layer

    def encode(self, encoder_input, encoder_mask):
        """
        Forward pass for the encoder

        Args:
            encoder_input (torch.Tensor): Input of encoder.
            encoder_mask (torch.Tensor): Source mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Pass the source tensor through the embedding and positional encoding
        x = self.encoder_position(self.encoder_embed(encoder_input))
        # Pass through the encoder
        return self.encoder(x, encoder_mask)

    def decode(self, encoder_output, encoder_mask, decoder_input, decoder_mask):
        """ 
        Forward pass for the decoder
        Args:
            encoder_output (torch.Tensor): Output from the encoder.
            encoder_mask (torch.Tensor): Encoder mask tensor.
            decoder_input (torch.Tensor): Input of decoder
            decoder_mask (torch.Tensor): Mask of the encoder.
        Returns:
            torch.Tensor: Output tensor.
        """
        #print(f'shape of decoder_input in model.decode: {decoder_input.shape}')
        x = self.decoder_position(self.decoder_embed(decoder_input))
        # Pass through the decoder
        return self.decoder(x, encoder_output, encoder_mask, decoder_mask)

    def projection(self, x):
        """
        Forward pass for the projection layer

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.projection_layer(x)


def build_transformer_model(config, source_vocab_size, target_vocab_size) -> Transformer:
    """
    Function to build the transformer model

    Args:
        config (_type_): _description_
    """
    #source_vocab_size = config['MODEL']['source_vocab_size']
    #target_vocab_size = config['MODEL']['target_vocab_size']
    source_sq_len = config['MODEL']['source_sq_len']
    target_sqe_len = config['MODEL']['target_sqe_len']
    embedding_dim = config['MODEL']['embedding_dim']
    num_heads = config['MODEL']['num_heads']
    num_layers = config['MODEL']['num_layers']
    dropout = config['MODEL']['dropout']
    ff_hidden_size = config['MODEL']['ff_hidden_size']

    # Create the input embedding layers
    encoder_embed = InputEmbedding(source_vocab_size, embedding_dim)
    decoder_embed = InputEmbedding(target_vocab_size, embedding_dim)

    # Position encoding layers
    encoder_position = PositionEncoding(source_sq_len, embedding_dim, dropout)
    decoder_position = PositionEncoding(target_sqe_len, embedding_dim, dropout)

    # Create the encoder and decoder
    encoder = Encoder(num_layers, embedding_dim,
                      num_heads, ff_hidden_size, dropout)
    decoder = Decoder(num_layers, embedding_dim,
                      num_heads, ff_hidden_size, dropout)

    # Create the projection layer
    projection_layer = ProjectionLayer(embedding_dim, target_vocab_size)

    # Create the transformer model
    model = Transformer(encoder, decoder, encoder_embed, decoder_embed,
                        encoder_position, decoder_position, projection_layer)

    # Initialize the weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model