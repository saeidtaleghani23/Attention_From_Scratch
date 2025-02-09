import pytest 
import torch
from model.Attention_model import (
    InputEmbedding, PositionEncoding, LayerNormalization,
    FeedForward, MultiHeadAttention, ResidualConnection, EncoderBlock, DecoderBlock,
    Encoder, Decoder, Transformer, ProjectionLayer
)

def test_input_embedding():
    vocab_size, d_model = 50000, 512
    embed = InputEmbedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (2, 10))
    output = embed(x)
    assert output.shape == (2, 10, d_model), "Output shape of InputEmbedding is not correct"

def test_position_encoding():
    max_seq_len, d_model = 100, 512
    pos_enc = PositionEncoding(max_seq_len, d_model )
    x = torch.rand(2, max_seq_len, d_model)
    output = pos_enc(x)
    assert output.shape == (2, max_seq_len, d_model), "Output shape of PositionEncoding not correct"

def test_layer_normalization():
    d_model = 512
    norm = LayerNormalization(d_model)
    x = torch.rand(2, 10, d_model)
    output = norm(x)
    assert output.shape == (2, 10, d_model), "Output shape of LayerNormalization not correct"

def test_feed_forward():
    d_model, d_ff = 512, 2048
    ff = FeedForward(d_model, d_ff)
    x = torch.rand(2, 10, d_model)
    output = ff(x)
    assert output.shape == (2, 10, d_model), "Output shape of FeedForward not correct"

def test_multi_head_attention():
    d_model, num_heads = 512, 8
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.rand(2, 10, d_model)
    output = mha(x, x, x)
    assert output.shape == (2, 10, d_model), "Output shape of MultiHeadAttention not correct" 

def test_residual_connection():
    drop = 0.2
    d_model = 512
    d_model, d_ff = 512, 2048
    ff = FeedForward(d_model, d_ff)
    rc = ResidualConnection(drop)
    x = torch.rand(2, 10, d_model)
    output = rc(x, ff)
    assert output.shape == (2, 10, d_model), "Output shape of ResidualConnection not correct"

def test_encoder_block():
    d_model, num_heads, d_ff , drop= 512, 8, 2048, 0.1
    eb = EncoderBlock(d_model, num_heads, d_ff, drop)
    x = torch.rand(2, 10, d_model)
    output = eb(x, mask=None)
    print(f'output.shape: {output.shape}')
    assert output.shape == (2, 10, d_model), "Output shape of EncoderBlock not correct"

def test_decoder_block():
    d_model, num_heads, d_ff = 512, 8, 2048
    db = DecoderBlock(d_model, num_heads, d_ff)
    x = torch.rand(2, 10, d_model)
    source_mask, target_mask = None, None
    output = db(x, x,  source_mask, target_mask )
    assert output.shape == (2, 10, d_model), "Output shape of DecoderBlock not correct"

def test_Encoder():
    layers, d_model, num_heads, d_ff, drop = 6, 512, 8, 2048, 0.2
    encoder = Encoder(layers, d_model, num_heads, d_ff, drop)
    x = torch.rand(2, 10, d_model)
    output = encoder(x, None)
    assert output.shape == (2, 10, d_model), "Output shape of Encoder not correct"

def test_Decoder():
    layers, d_model, num_heads, d_ff, drop = 6, 512, 8, 2048, 0.2
    decoder = Decoder(layers, d_model, num_heads, d_ff, drop)
    x = torch.rand(2, 10, d_model)
    source_mask, target_mask = None, None
    output = decoder(x, x, source_mask, target_mask)
    assert output.shape == (2, 10, d_model), "Output shape of Decoder not correct"

def test_ProjectionLayer():
    layers, d_model, num_heads, d_ff, drop = 6, 512, 8, 2048, 0.2
    decoder = Decoder(layers, d_model, num_heads, d_ff, drop)
    x = torch.rand(2, 10, d_model)
    source_mask, target_mask = None, None
    output = decoder(x, x, source_mask, target_mask)
    projection_layer = ProjectionLayer(embed_size=d_model, vocab_size=50000)
    project_output = projection_layer(output)



def test_Transformer():
    # Define necessary components for the transformer
    layers, embed_size, num_heads, d_ff, drop = 6, 512, 8, 2048, 0.2
    encoder_embed = InputEmbedding(vocab_size=50000, embed_size=embed_size)
    decoder_embed = InputEmbedding(vocab_size=50000, embed_size=embed_size)

    encoder_position = PositionEncoding(max_seq_len=100, embedding_dim=embed_size, drop = drop)
    decoder_position = PositionEncoding(max_seq_len=100, embedding_dim=embed_size, drop = drop)

    encoder = Encoder(layers= layers, embed_size=embed_size, heads=num_heads, ff_hidden_size=d_ff, drop=drop)
    decoder = Decoder(layers= layers, embed_size=embed_size, heads=num_heads, ff_hidden_size=d_ff, drop=drop)
    projection_layer = ProjectionLayer(embed_size=embed_size, vocab_size=50000)

    transformer = Transformer(encoder, 
                              decoder, 
                              encoder_embed, 
                              decoder_embed, 
                              encoder_position, 
                              decoder_position, 
                              projection_layer)

    # Test the transformer
    encoder_input = torch.randint(0, 50000, (2, 10))

    encoder_output = transformer.encode(source = encoder_input, source_mask = None)
    decoder_output = transformer.decode(target = torch.randint(0, 50000, (2, 10)), 
                                        encoder_output = encoder_output, 
                                        source_mask = None, 
                                        target_mask = None)
    project_output = transformer.projection_layer(decoder_output)
if __name__ == "__main__":
    pytest.main()