from model.Attention_model import build_transformer_model
import time
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import os
from sklearn.metrics import accuracy_score  # type: ignore
from util import get_dataset, get_weights, causal_mask
import numpy as np  # type: ignore
import wandb  # type: ignore
import yaml
from tqdm import tqdm  # type: ignore


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

        prob = model.project(out[:, -1]) # (batch, voc_size) --> (1, 10000)
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


def run_val_epoch(model,  val_dataloader,loss_function, device, encoder_tokenizer, decoder_tokenizer, max_seq_len, results, epoch,prefix = 'val'):

    model= model.to(device)
    model.eval()
    running_accuracy = []
    running_loss = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device) # (1, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (1, 1, 1, seq_len)
            label = batch["label"].to(device)  # (1, max_Seq_len)
            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"
            # Use greedy decoding to generate the predicted sequences token IDs
            predicted_tokens = greedy_decode(model, encoder_input, encoder_mask, encoder_tokenizer, decoder_tokenizer, max_seq_len, device)

            # Calculate loss for each batch (like in the training phase)
            encoder_output = model.encode(encoder_input, encoder_mask)  # (1, max_Seq_len, embedding_dim)
            decoder_output = model.decoder(
            encoder_output, encoder_mask, predicted_tokens, encoder_mask  # Use predicted tokens as decoder input
        )
            projection_output = model.projection(decoder_output)  # (1, max_Seq_len, target_vocab_size)

            # Compute loss (cross entropy loss)
            loss = loss_function(
                projection_output.view(-1, decoder_tokenizer.get_vocab_size()),  # Flatten the output
                label.view(-1)  # Flatten the labels as well
            )

            # Show the loss for this batch
            running_loss.append(loss.item())

            # Calculate accuracy for each sequence
            pad_token_id = decoder_tokenizer.token_to_id("[PAD]")
            non_pad_mask = label != pad_token_id  # Mask out padding tokens

            # Avoid division by zero
            if non_pad_mask.sum() > 0:  
                correct_predictions = (predicted_tokens == label) & non_pad_mask  # Compare predicted and actual tokens
                accuracy = correct_predictions.sum() / non_pad_mask.sum()  # Scalar accuracy for the batch
                running_accuracy.append(accuracy.item())
            else:
                running_accuracy.append(0)  # If no non-pad tokens, append 0 accuracy


    # Calculate average loss and accuracy
    avg_loss = np.mean(running_loss)
    avg_accuracy = np.mean(running_accuracy)
    
    # Store results in dictionary 
    results[prefix + " loss"] = avg_loss
    results[prefix + " accuracy"] = avg_accuracy

    # log the loss and accuracy value to WandB
    wandb.log({f'{prefix} loss': avg_loss, "epoch": epoch})
    wandb.log({f'{prefix} accuracy': avg_accuracy * 100 , "epoch": epoch})

    return 

      
def run_train_epoch(model,optimizer,train_dataloader,loss_function, device,results, encoder_tokenizer, epoch):
    
    model = model.to(device)
    model.train()
    running_loss = []
    running_accuracy = []
    batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
    for batch in batch_iterator:
        encoder_input = batch["encoder_input"].to(device)  # (Batch, max_Seq_len)
        decoder_input = batch["decoder_input"].to(device)  # (Batch, max_Seq_len)
        encoder_mask = batch["encoder_mask"].to(device)  # (Batch, 1, 1, max_Seq_len)
        decoder_mask = batch["decoder_mask"].to(device)  # (Batch, 1, 1, max_Seq_len)
        label = batch["label"].to(device)  # (Batch, max_Seq_len)

        # Output of the model
        # (Batch, Max_Seq_len, embedding_dim)
        encoder_output = model.encode(encoder_input, encoder_mask)

        # (Batch, Max_Seq_len, embedding_dim)
        decoder_output = model.decoder(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )

        # (Batch, Max_Seq_len, target_vocab_size)
        projection_output = model.projection(decoder_output)

        # Compute loss for each batch.
        # first  (Batch, Max_Seq_len, target_vocab_size) --> (Batch * Max_Seq_len, target_vocab_size)
        loss = loss_function(
            projection_output.view(-1, encoder_tokenizer.get_vocab_size()),
            label.view(-1),
        )


        # Show the lost on the progress bar
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # batch loss value
        running_loss.append(loss.item())

        # batch Accuracy
        predicted_tokens = torch.argmax(projection_output, dim=-1)  # Shape: (Batch, Max_Seq_len)
        # Mask Padding Tokens in Labels
        pad_token_id = encoder_tokenizer.token_to_id("[PAD]")
        non_pad_mask = label != pad_token_id  # Shape: (Batch, Max_Seq_len)
        # Calculate Accuracy
        correct_predictions = (predicted_tokens == label) & non_pad_mask  # Shape: (Batch, Max_Seq_len)
        accuracy = correct_predictions.sum() / non_pad_mask.sum()  # Scalar value
        running_accuracy.append(accuracy)

    # Loss value after each epoch
    results["train loss"].append(np.mean(running_loss))
    results["train accuracy"].append(np.mean(running_loss))

    # log the loss value
    wandb.log({"Train loss": np.mean(running_loss), "epoch": epoch})


def train_model(
    initial_epoch,
    epochs,
    model,
    optimizer,
    train_loader,
    loss_function,
    device,
    result_path,
    encoder_tokenizer, 
    decoder_tokenizer,
    max_seq_len,
    validation_loader=None,
):

    # -- save all results
    checkpoint_file_results = os.path.join(result_path, ("All_results.pt"))
    # -- save the best result based on validation accuracy
    checkpoint_file_best_result = os.path.join(result_path, ("BestResult.pt"))

    # -- send model on the device
    model = model.to(device)
    to_track = ["epoch", "train loss", "train accuracy"]

    # -- There is Validation loader?
    if validation_loader is not None:
        to_track.append("val accuracy")
        to_track.append("val loss")

    results = {}

    # -- Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    Best_validation_loss = np.inf

    # -- Train model
    print("Training begins...\n")

    for epoch in range(initial_epoch, epochs):
        # -- set the model on train
        model.train()
        # -- Train for one epoch
        run_train_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            device,
            results,
            encoder_tokenizer,
            epoch,
        )

        # -- Save epoch and processing time
        results["epoch"].append(epoch)

        #   ******  Validating  ******
        if validation_loader is not None:
            run_val_epoch(
                    model,
                    validation_loader,
                    loss_function,
                    device,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    max_seq_len,
                    results,
                    epoch,
                    prefix = 'val'
                )
            # save the model based on the validation loss
            if results["val loss"][-1] < Best_validation_loss:
                print(
                    "\nEpoch: {}   train loss: {:.2f}   train accuracy: {:.2f} val loss: {:.2f}   val accuracy: {:.2f}".format(
                        epoch,
                        results["train loss"][-1],
                        results["train accuracy"][-1],
                        results["val loss"][-1],
                        results["val accuracy"][-1],
                    )
                )
                Best_validation_loss = results["val loss"][-1]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_file_best_result,
                )
        
    #  Save all recorded results 
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "results": results,
        },
        checkpoint_file_results,
    )


if __name__ == "__main__":
    # Read the config file
    config_path = os.path.join(os.getcwd(), "config", "config.yml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    #
    source_language = config["DATASET"]["source_lang"]
    target_language = config["DATASET"]["target_lang"]
    model_name = config["BENCHMARK"]["model_name"]

    # Initialize W&B
    # Authenticate with your API key
    wandb.login(key="4dd27c7624f2ab82554553d3e872b47dcaa05780")
    wandb.init(
        project=f"translation from {source_language} to {target_language}",  # Name of your project
        config={
            "learning_rate": config["TRAIN"]["lr"],
            "batch_size": config["TRAIN"]["batch_size"],
            "epochs": config["TRAIN"]["epochs"],
            "model": "Transformer",
        },
    )

    Result_Directory = os.path.join(
        config["BENCHMARK"]["results_path"], config["BENCHMARK"]["model_name"]
    )
    os.makedirs(Result_Directory, exist_ok=True)

    # getting the dataloaders, and tokenizers
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        encoder_tokenizer,
        decoder_tokenizer,
    ) = get_dataset(config)

    # get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer_model(config)
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAIN"]["lr"], eps=1e-9)

    

    result_path = os.path.join(
        config["BENCHMARK"]["model_folder"],
        f"{model_name}_{source_language}_{target_language}",
    )
    os.makedirs(result_path, exist_ok=True)

    initial_epoch = 0
    global_step = 0
    if config["TRAIN"]["preload"]:
        saved_model_path = os.path.join(result_path, "BestResult.pt")
        # Assuming you have defined your model and optimizer
        checkpoint = torch.load(saved_model_path)
        # Load the model state
        model.load_state_dict(checkpoint["model_state_dict"])
        # Load the optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Optionally, load the epoch and results for tracking
        initial_epoch = checkpoint["epoch"] + 1

    loss_function = nn.CrossEntropyLoss(
        ignore_index=encoder_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    Start = time.time()

    # -- Train the model
    train_model(
        initial_epoch,
        config["TRAIN"]["epochs"],
        model,
        optimizer,
        train_dataloader,
        loss_function,
        device,
        result_path,
        encoder_tokenizer,
        decoder_tokenizer,
        max_seq_len= config["MODEL"]["source_sq_len"],
        validation_loader=val_dataloader,
    )

    End = time.time()
    Diff_hrs = (End - Start) / 3600
    print("***********      End of Training        **************")
    print("\n It took: {:.3f} hours".format(Diff_hrs))
