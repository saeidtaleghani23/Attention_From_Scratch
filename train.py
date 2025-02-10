from model.Attention_model import build_transformer_model
import time
import torch # type: ignore
import torch.nn as nn # type: ignore 
import os
from sklearn.metrics import accuracy_score # type: ignore
from util import get_dataset, get_weights 
import numpy as np # type: ignore
import wandb
import yaml
from tqdm import tqdm # type: ignore


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_epoch_val(model, optimizer, data_loader, loss_function, device, results, score_functions, epoch):
    model = model.to(device)
    running_loss = []
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(data_loader, desc= f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (Batch, max_Seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, max_Seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (Batch, 1, 1, max_Seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, 1, max_Seq_len)
            label = batch['label'].to(device) #(Batch, max_Seq_len)
            # Output of the model
            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, Max_Seq_len, embedding_dim)
            decoder_output = model.decoder(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, Max_Seq_len, embedding_dim)
            projection_output = model.projection(decoder_output) # (Batch, Max_Seq_len, target_vocab_size)

            # Compute loss for each batch.
            # first  (Batch, Max_Seq_len, target_vocab_size) --> (Batch * Max_Seq_len, target_vocab_size) 
            loss = loss_function(projection_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))

            # Show the lost on the progress bar
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            # loss value
            running_loss.append(loss.item())
        # Loss value after each epoch
        results["train loss"].append(np.mean(running_loss))
        # log the loss value
        wandb.log({"Val loss": np.mean(running_loss), "epoch": epoch})
   
    


def run_epoch_train(model, optimizer, data_loader, loss_function, device, results, score_functions, epoch):
    model = model.to(device)
    model.train()
    running_loss = []
    batch_iterator = tqdm(data_loader, desc= f'Processing epoch {epoch:02d}')
    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device) # (Batch, max_Seq_len)
        decoder_input = batch['decoder_input'].to(device) # (Batch, max_Seq_len)
        encoder_mask = batch['encoder_mask'].to(device) # (Batch, 1, 1, max_Seq_len)
        decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, 1, max_Seq_len)
        label = batch['label'].to(device) #(Batch, max_Seq_len)
        # Output of the model
        encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, Max_Seq_len, embedding_dim)
        decoder_output = model.decoder(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, Max_Seq_len, embedding_dim)
        projection_output = model.projection(decoder_output) # (Batch, Max_Seq_len, target_vocab_size)

        # Compute loss for each batch.
        # first  (Batch, Max_Seq_len, target_vocab_size) --> (Batch * Max_Seq_len, target_vocab_size) 
        loss = loss_function(projection_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))

        # Show the lost on the progress bar
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        # Back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # loss value
        running_loss.append(loss.item())
    # Loss value after each epoch
    results["train loss"].append(np.mean(running_loss))

    # log the loss value
    wandb.log({"Train loss": np.mean(running_loss), "epoch": epoch})

def train_model(initial_epoch,
                epochs,
                model,
                optimizer,
                train_loader,
                loss_func,
                score_functions,
                device,
                result_path,
                validation_loader=None,
                test_loader=None):

    # -- save all results
    checkpoint_file_results = os.path.join(
        result_path, ('All_results.pt'))
    # -- save the best result based on validation accuracy
    checkpoint_file_best_result = os.path.join(
        result_path, ('BestResult.pt'))

    # -- send model on the device
    model = model.to(device)
    to_track = ["epoch", "total time", "train loss"]

    # -- There is Validation loader?
    if validation_loader is not None:
        to_track.append("validation loss")

    # -- There is test loader ?
    if test_loader is not None:
        to_track.append("test loss")

    total_train_time = 0
    results = {}

    # -- Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    Best_validation_loss = np.inf

    # -- Train model
    print('Training begins...\n')

    for epoch in range(initial_epoch, epochs):
        # -- set the model on train
        model.train()
        # -- Train for one epoch
        total_train_time += run_epoch(model, optimizer, train_loader,
                                      loss_func, device, results,
                                      score_functions, epoch, prefix="train", desc="Training")

        # -- Save epoch and processing time
        results["epoch"].append(epoch)
        results["total time"].append(total_train_time)

        #   ******  Validating  ******
        if validation_loader is not None:
            model = model.eval()  # Set the model to "evaluation" mode
            with torch.no_grad():
                run_epoch(model, optimizer, validation_loader,
                          loss_func, device, results,
                          score_functions, prefix="validation", desc="Validating")

        #   ******  Testing  ******
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader,
                          loss_func, device, results,
                          score_functions, prefix="test", desc="Testing")

        #   ******  Save results of each epoch  ******
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results
        }, checkpoint_file_results)
        
        # save the model based on the validation loss
        if results['validation loss'][-1] < Best_validation_loss:
            print('\nEpoch: {}   Training loss: {:.2f}   best Val loss: {:.2f}   Test loss: {:.2f}'
                  .format(epoch, results['train loss'][-1], results['validation loss'][-1], results['test loss'][-1]))
            Best_validation_loss = results['validation loss'][-1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_file_best_result)


if __name__ == '__main__':
    # Read the config file
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize W&B
    # Authenticate with your API key
    wandb.login(key="4dd27c7624f2ab82554553d3e872b47dcaa05780")
    wandb.init(
        project="translation",  # Name of your project
        config={
            "learning_rate": config['TRAIN']['lr'],
            "batch_size": config['TRAIN']['batch_size'],
            "epochs": config['TRAIN']['epochs'],
            "model": 'english_to_french',
        })

    Result_Directory = os.path.join(
        config['BENCHMARK']['results_path'], config['BENCHMARK']['model_name'])
    os.makedirs(Result_Directory, exist_ok=True)

    score_functions = {"Accuracy": accuracy_score}
    # getting the dataloaders
    train_dataloader, val_dataloader, test_dataloader, source_tokenizer, target_tokenizer = get_dataset(
        config)
    # get model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_transformer_model(config)
    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['TRAIN']['lr'], eps=1e-9)
    
    #
    source_language = config['DATASET']['source_lang']
    target_language = config['DATASET']['target_lang']
    model_name = config['BENCHMARK']['model_name']

    result_path = os.path.join(config['BENCHMARK']['model_folder'], f'{model_name}_{source_language}_{target_language}')
    os.makedirs(result_path, exist_ok=True)


    initial_epoch= 0
    global_step = 0
    if config['TRAIN']['preload']:
        saved_model_path = os.path.join(result_path,'BestResult.pt')
        # Assuming you have defined your model and optimizer
        checkpoint = torch.load(saved_model_path)
        # Load the model state
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load the optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Optionally, load the epoch and results for tracking
        initial_epoch = checkpoint['epoch'] +1
    
    loss_function = nn.CrossEntropyLoss(ignore_index = source_tokenizer.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

    Start = time.time()

    # -- Train the model
    train_model(initial_epoch,
        config["TRAIN"]["epochs"],
        model,
        optimizer,
        train_dataloader,
        loss_function,
        score_functions,
        device,
        result_path,
        validation_loader=val_dataloader,
        test_loader=test_dataloader,
    )

    End = time.time()
    Diff_hrs = (End - Start) / 3600
    print("***********      End of Training        **************")
    print("\n It took: {:.3f} hours".format(Diff_hrs))