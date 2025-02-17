from flask import Flask, request, render_template, jsonify # type: ignore
import os
import yaml
import torch # type: ignore
from tokenizers import Tokenizer # type: ignore
from model.Attention_model import build_transformer_model
from test_model import get_trained_model, greedy_decode
from util import causal_mask

app = Flask(__name__)

def greedy_encode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(encoder_input, encoder_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # build mask for target
        # Causal mask for the decoder (lower triangular matrix)
        # encoder_input.size(1) # max_seq_len
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device) # (1, seq_len, seq_len)
        # calculate output
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #    
        prob = model.projection(out[:, -1]) # (batch, voc_size) --> (1, 10000)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
    
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        # Get the sentence from the form
        sentence = request.form['sentence']

        # Read config
        config_path = os.path.join(os.getcwd(), "config", "config.yml")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Load the trained model and tokenizers using the imported function
        model, encoder_tokenizer, decoder_tokenizer = get_trained_model(config)

        # get SOS, EOS, PAD tokens IDs
        sos_token = torch.tensor([encoder_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        eos_token = torch.tensor([encoder_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        pad_token = torch.tensor([encoder_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

        # Convert input sentence into token IDs
        encoder_input_tokens = encoder_tokenizer.encode(sentence).ids


        # We need to add SOS and EOS tokens into encoder tokens and pad the sentence to the max_seq_len
        encoder_padding_tokens = config['MODEL']['source_sq_len'] - \
            len(encoder_input_tokens) - 2
        
        encoder_input = torch.cat([sos_token,
                                   torch.tensor(
                                       encoder_input_tokens, dtype=torch.int64),
                                   eos_token,
                                   torch.tensor([pad_token] * encoder_padding_tokens, dtype=torch.int64)])
        
        encoder_input= encoder_input.unsqueeze(0)
        
        # Create encoder mask that 1 shows not padded token ids and 0 shows padded token ids
        encoder_mask = (encoder_input != pad_token).unsqueeze(
            0).unsqueeze(0).int()  # (1, 1, max_seq_len)

        # model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device
        max_len = config['MODEL']['target_sqe_len']
        translated_tokens = greedy_decode(model, encoder_input, encoder_mask, encoder_tokenizer, decoder_tokenizer, max_len=max_len, device="cpu")

        # Decode translated tokens into text
        translated_text = decoder_tokenizer.decode(translated_tokens.detach().cpu().numpy())

        return render_template('index.html', original=sentence, translated=translated_text)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
