import torch # type: ignore
from util import causal_mask
from model.Attention_model import build_transformer_model
from tokenizers import Tokenizer # type: ignore
import wandb  # type: ignore
import numpy as np # type: ignore
from tqdm import tqdm # type: ignore
import torchmetrics # type: ignore
import os
import yaml
from torch.utils.data import DataLoader # type: ignore
import warnings
warnings.simplefilter("ignore", FutureWarning)

def get_trained_model(config):
    # generate a model with the randomly initialized weights 
    encoder_tokenizer_path = os.getcwd()+ f"/tokenizer/tokenizer_{config['DATASET']['source_lang']}.json"
    encoder_tokenizer = Tokenizer.from_file(encoder_tokenizer_path)

    decoder_tokenizer_path = os.getcwd()+ f"/tokenizer/tokenizer_{config['DATASET']['target_lang']}.json"
    decoder_tokenizer = Tokenizer.from_file(decoder_tokenizer_path)

    model = build_transformer_model(config,  encoder_tokenizer.get_vocab_size(), decoder_tokenizer.get_vocab_size())

    # get the trained weights
    weights = os.getcwd()+ f"/trained_model/weights/Transformer_{config['DATASET']['source_lang']}_{config['DATASET']['target_lang']}"
    # Load the checkpoint
    checkpoint = torch.load(weights + '/BestResult.pt', map_location=torch.device('cpu'))  # Use 'cuda' if running on GPU

    # Load the saved weights into the new model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode (important for inference)
    model.eval()

    print("[INFO] Model weights loaded successfully!")
    
    return model, encoder_tokenizer, decoder_tokenizer 


def greedy_decode(
    model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
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

        # get next token
        # Python Note:
        # out[:, -1] is the embedding of the last token from the decoder output.
        # If out.shape = (1, seq_len, embed_dim), then out[:, -1].shape = (1, embed_dim).
        # ":" means "select all batches" (in this case, batch size is 1).
        # "-1" refers to the last token along the sequence dimension (the final token in the sequence).
        # Since there's no index for the embed_dim dimension, all 512 values are selected.
        # In other words, out[:, -1] = out[:, -1, :]

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

def test(model, test_dataloader, device, encoder_tokenizer, decoder_tokenizer, max_seq_len, prefix = 'test', print_samples = 10):
    # random samples
    random_samples_idx = np.random.randint(0, len(test_dataloader), print_samples).tolist()
    model= model.to(device)
    model.eval()
    running_accuracy = []
    predicted_sentence = [] 
    expected_sentence= []
    source_sentence = []
    with torch.no_grad():
        batch_iterator = tqdm(test_dataloader, desc=f" Processing {prefix} data")
        count = 0
        for batch in  batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (1, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (1, 1, 1, seq_len)
            label = batch["label"].to(device)  # (1, max_Seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (1, 1, max_Seq_len, max_Seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for test"
            # Use greedy decoding to generate the predicted sequences token IDs  
            predicted_tokens = greedy_decode(model, encoder_input, encoder_mask, encoder_tokenizer, decoder_tokenizer, max_seq_len, device)
            model_out_text = decoder_tokenizer.decode(predicted_tokens.detach().cpu().numpy())

            # save the sentences 
            source_sentence.append(batch['source_text'][0])
            expected_sentence.append(batch['target_text'][0])
            predicted_sentence.append(model_out_text)

            # Check if count is in the selected indices
            if count in random_samples_idx:
                print('*'*100)
                print(f"input sentence: {batch['source_text'][0]}")
                print(f"output sentence: {batch['target_text'][0]}")
                print(f"predicted sentence: {model_out_text}")
            count+=1 # Increment count
    
    # calculate the performance of the model on the test dataset

    # # Compute Char Error Rate (CER)
    metric1 = torchmetrics.CharErrorRate()
    CharErrorMetric = metric1(predicted_sentence, expected_sentence)

    # Compute Word Error Rate (WER)
    metric2 = torchmetrics.WordErrorRate()
    WordErrorRate = metric2(predicted_sentence, expected_sentence)

    # Compute BLEU Score
    metric3 = torchmetrics.BLEUScore()
    BLEUScore = metric3(predicted_sentence, expected_sentence)

    return CharErrorMetric, WordErrorRate, BLEUScore

if __name__=='__main__':
    # read the config 
    config_path = os.path.join(os.getcwd(), "config", "config.yml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # call the trained model
    print('[INFO] reading the trained model')
    trained_model, encoder_tokenizer, decoder_tokenizer  = get_trained_model(config)
    
    # Load the saved dataloader components
    print('[INFO] reading test data loader')
    file_path='./dataloaders/test_dataloader.pth'
    checkpoint = torch.load(file_path)
    # Extract the test_dataset
    test_dataset = checkpoint['test_dataset']
    # Recreate the test DataLoader with batch_size = 1 and shuffle = False
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('[INFO] test the model performance')
    CharErrorMetric, WordErrorRate, BLEUScore = test(trained_model,
                                                     test_dataloader, 
                                                     device, 
                                                     encoder_tokenizer, 
                                                     decoder_tokenizer, 
                                                     max_seq_len = config['MODEL']['target_sqe_len'],
                                                     prefix = 'test',
                                                     print_samples = 10)
    print(f"CharErrorMetric:{CharErrorMetric}\nWordErrorRate:{WordErrorRate}\nBLEUScore:{BLEUScore}")
    

   



