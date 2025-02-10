from model.Attention_model import build_transformer_model
import time
import torch
import os
from sklearn.metrics import accuracy_score
from util import get_dataset
import numpy as np
import wandb
import yaml


def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj


def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_functions, prefix="", desc=None):

    model = model.to(device)
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in (data_loader):
        # -- Move the batch to the device we are using.
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        if prefix == "validation" or prefix == "test":
            inputs.requires_grad_(False)  # Ensure inputs don't track gradients
            labels.requires_grad_(False)  # Ensure labels don't track gradients

        # -- Output of the model
        y_hat = model(inputs)

        # -- Compute loss.
        loss = loss_func(y_hat, labels)

        # -- Training?
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # -- Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_functions) > 0 and isinstance(labels, torch.Tensor):
            # -- moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            # -- add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    # -- end of one training epoch
    end = time.time()

    y_pred = np.asarray(y_pred)
    # We have a classification problem, convert to labels
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    # Else, we assume we are working on a regression problem

    results[prefix + " loss"].append(np.mean(running_loss))
    for name, score_func in score_functions.items():
        try:
            results[prefix + " " + name].append(score_func(y_true, y_pred))
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end-start  # time spent on epoch


def train_model(epochs,
                model,
                optimizer,
                train_loader,
                loss_func,
                score_functions,
                result_path,
                patch_size,
                validation_loader=None,
                test_loader=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- Create Result file
    if os.path.exists(result_path) is not True:
        os.mkdir(result_path)

    # -- save all results
    checkpoint_file_results = os.path.join(
        result_path, ('All_results.pt'))
    # -- save the best result based on validation accuracy
    checkpoint_file_best_result = os.path.join(
        result_path, ('BestResult.pt'))

    # -- send model on the device
    model = model.to(device)
    to_track = ["epoch", "total time", "train Accuracy", "train loss"]

    # -- There is Validation loader?
    if validation_loader is not None:
        to_track.append("validation Accuracy")
        to_track.append("validation loss")

    # -- There is test loader ?
    if test_loader is not None:
        to_track.append("test Accuracy")
        to_track.append("test loss")

    total_train_time = 0
    results = {}

    # -- Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    Best_validation_Accuracy = 0.0

    # -- Train model
    print('Training begins...\n')

    for epoch in range(epochs):
        # -- set the model on train
        model = model.train()
        # -- Train for one epoch
        total_train_time += run_epoch(model, optimizer, train_loader,
                                      loss_func, device, results,
                                      score_functions, prefix="train", desc="Training")

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
        # show the progress and metrics
        print('\nEpoch: {}   Training accuracy: {:.2f}   Validation accuracy: {:.2f}   Test Accuracy: {:.2f}'
              .format(epoch, results['train Accuracy'][-1]*100, results['validation Accuracy'][-1]*100, results['test Accuracy'][-1]*100))
        # save the model based on the validation accuracy
        if results['validation Accuracy'][-1] > Best_validation_Accuracy:
            print('\nEpoch: {}   Training accuracy: {:.2f}   best Val accuracy: {:.2f}   Test Accuracy: {:.2f}'
                  .format(epoch, results['train Accuracy'][-1]*100, results['validation Accuracy'][-1]*100, results['test Accuracy'][-1]*100))
            Best_validation_Accuracy = results['validation Accuracy'][-1]
            best_result = {}
            best_result["epoch"] = []
            best_result["train accuracy"] = []
            best_result["validation accuracy"] = []
            best_result["test accuracy"] = []

            best_result["epoch"].append(epoch)
            best_result["train accuracy"].append(results['train Accuracy'][-1])
            best_result["validation accuracy"].append(
                results['validation Accuracy'][-1])
            best_result["test accuracy"].append(results['test Accuracy'][-1])

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': best_result
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
    from model.Attention_model import build_transformer_model
    model = build_transformer_model(config)
    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['TRAIN']['lr'], eps=1e-9)
