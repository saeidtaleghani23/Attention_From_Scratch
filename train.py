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
from datetime import datetime

      
def run_one_epoch(model,optimizer,dataloader,loss_function, device,results, encoder_tokenizer, epoch, prefix= 'train'):
    
    model = model.to(device)
    
    running_loss = []
    running_accuracy = []
    batch_iterator = tqdm(dataloader, desc=f" {prefix} Processing epoch {epoch:02d}")
    for batch in batch_iterator:
        encoder_input = batch["encoder_input"].to(device)  # (Batch, max_Seq_len)
        decoder_input = batch["decoder_input"].to(device)  # (Batch, max_Seq_len)
        encoder_mask = batch["encoder_mask"].to(device)  # (Batch, 1, 1, max_Seq_len)
        decoder_mask = batch["decoder_mask"].to(device)  # (Batch, 1, max_Seq_len, max_Seq_len)
        label = batch["label"].to(device)  # (Batch, max_Seq_len)
        if prefix == 'train':
            model.train()
        elif prefix == 'val':
            model.eval()
            encoder_input.requires_grad_(False)
            decoder_input.requires_grad_(False)
            encoder_mask.requires_grad_(False)
            decoder_mask.requires_grad_(False)
            label.requires_grad_(False)

        # Output of the model
        # (Batch, Max_Seq_len, embedding_dim)
        encoder_output = model.encode(encoder_input, encoder_mask)
        # (Batch, Max_Seq_len, embedding_dim)
        decoder_output = model.decode(
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
        # batch loss value
        running_loss.append(loss.item())

        # Show the lost on the progress bar
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Back propagation
        if  prefix == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # batch Accuracy
        predicted_tokens = torch.argmax(projection_output, dim=-1)  # Shape: (Batch, Max_Seq_len)
        # Mask Padding Tokens in Labels
        pad_token_id = encoder_tokenizer.token_to_id("[PAD]")
        non_pad_mask = label != pad_token_id  # Shape: (Batch, Max_Seq_len)
        # Calculate Accuracy
        correct_predictions = (predicted_tokens == label) & non_pad_mask  # Shape: (Batch, Max_Seq_len)
        accuracy = correct_predictions.sum().item() / non_pad_mask.sum().item()  # Scalar value
        running_accuracy.append(accuracy)

    
    # Calculate average loss and accuracy for a epoch
    epoch_loss = np.mean(running_loss)
    epoch_accuracy = np.mean(running_accuracy)

    results[prefix + " loss"].append(epoch_loss)
    results[prefix + " accuracy"].append(epoch_accuracy)

    # log the loss value
    wandb.log({f"{prefix} loss": epoch_loss, "epoch": epoch})


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
        run_one_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            device,
            results,
            encoder_tokenizer,
            epoch,
            prefix = 'train'
        )

        # -- Save epoch and processing time
        results["epoch"].append(epoch)

        #   ******  Validating  ******
        if validation_loader is not None:
            run_one_epoch(
                    model,
                    optimizer,
                    validation_loader,
                    loss_function,
                    device,
                    results,
                    encoder_tokenizer,
                    epoch,
                    prefix = 'val'
                )
            # save the model based on the validation loss
            if results['val loss'][-1] < Best_validation_loss:
                print(
                    "\nEpoch: {}   train loss: {:.4f}   train accuracy: {:.2f}% val loss: {:.4f}   val accuracy: {:.2f}%".format(
                        epoch,
                        results['train loss'][-1],
                        results['train accuracy'][-1]*100,
                        results['val loss'][-1],
                        results['val accuracy'][-1]*100,
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
        project=f"translation from {source_language} to {target_language}",  # name of your project
        name= f"{source_language}_to_{target_language}_run_{datetime.today().strftime('%Y-%m-%d_%H:%M')}", # name of run
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


    # save the test dataloader
    os.makedirs('./dataloaders', exist_ok=True)
    file_path='./dataloaders/test_dataloader.pth'
    # Save the datasets
    torch.save({
        'test_dataset': test_dataloader.dataset,
    }, file_path)
   
    # get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{device} device is being used')
    #print(f' source vocab size:{encoder_tokenizer.get_vocab_size()}\ntarget vocab size:{decoder_tokenizer.get_vocab_size()}')
    model = build_transformer_model(config,  encoder_tokenizer.get_vocab_size(), decoder_tokenizer.get_vocab_size())
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
