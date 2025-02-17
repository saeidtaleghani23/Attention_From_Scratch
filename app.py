from flask import Flask, request, jsonify # type: ignore
import os
import yaml
import torch # type: ignore
from tokenizers import Tokenizer # type: ignore
from model.Attention_model import build_transformer_model
from test import get_trained_model, greedy_decode  # Importing from test.py
from util import causal_mask  # Assuming this is from the 'util' module in your project

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        sentence = data['sentence']

        # Read config
        config_path = os.path.join(os.getcwd(), "config", "config.yml")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Load the trained model and tokenizers using imported function
        model, encoder_tokenizer, decoder_tokenizer = get_trained_model(config)

        # Tokenize input sentence using the tokenizer for the source language
        inputs = encoder_tokenizer.encode(sentence, return_tensors="pt")

        # Perform translation using greedy decoding (imported from test.py)
        translated_tokens = greedy_decode(model, inputs, None, encoder_tokenizer, decoder_tokenizer, max_len=50, device="cpu")

        # Decode translated tokens into text
        translated_text = decoder_tokenizer.decode(translated_tokens.detach().cpu().numpy())

        return jsonify({'translated_text': translated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
