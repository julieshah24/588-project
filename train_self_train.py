import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from text_encoders import RoBERTaEncoder
from datasets import SpamDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse


def update_labels_with_predictions(model, dataset, device="cpu", batch_size=32):
    """
    Updates the 'Spam/Ham' labels in the dataset with predictions from the provided model.
    Args:
    model (torch.nn.Module): The model to use for generating predictions.
    dataset (torch.utils.data.Dataset): The dataset whose labels are to be updated.
    device (str): Device to use for computations ('cpu' or 'cuda').
    batch_size (int): Number of samples per batch for processing.

    Returns:
    None: The dataset is modified in-place.
    """

    # Ensure model is in evaluation mode
    # Iterate over each item in the dataset using a simple loop
    for i in range(len(dataset)):
        data = dataset[i]  # Get single data point from the dataset

        # Prepare message for prediction
        message = data["Message"]

        # Generate prediction
        with torch.no_grad():
            outputs = model.predict_spam([message])
            _, predicted_label = torch.max(outputs, 1)

        # Update the label in the dataset
        # Since we are processing one at a time, we update directly
        dataset.data.loc[i, "Spam/Ham"] = predicted_label.item()

    # Set model back to training mode if necessary
    return dataset


def split_dataset(dataset, num_splits):
    split_size = len(dataset) // num_splits
    sub_datasets = []
    start_idx = 0
    for _ in range(num_splits):
        end_idx = min(start_idx + split_size, len(dataset))
        indices = list(range(start_idx, end_idx))
        sub_datasets.append(Subset(dataset, indices))
        start_idx = end_idx
    return sub_datasets


def combine_datasets(dataset_list):
    return ConcatDataset(dataset_list)


def self_train(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    max_iterations=1000,
    log_interval=10,
    eval_interval=100,
):
    iteration = 0
    loss_function = CrossEntropyLoss()

    while iteration < max_iterations:
        for data in train_loader:
            if iteration >= max_iterations:
                break

            message = data["Message"]
            
            # Predict labels instead of using true labels
            with torch.no_grad():
                pseudo_label = model.predict_spam(message).max(dim=1)[1]

            optimizer.zero_grad()
            prediction = model.predict_spam(message)
            loss = loss_function(prediction, pseudo_label)
            loss.backward()
            optimizer.step()

            if iteration % eval_interval == 0:
                test_accuracy = evaluate(encoder.model, test_loader)
                print(
                    f"Iteration: {iteration}, Loss: {loss.item()}, Test Accuracy: {test_accuracy}%"
                )
            elif iteration % log_interval == 0:
                print(f"Iteration: {iteration}, Loss: {loss.item()}")

            iteration += 1

def train_model(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    max_iterations=1000,
    log_interval=10,
    eval_interval=100,
):
    iteration = 0
    loss_function = CrossEntropyLoss()

    while iteration < max_iterations:
        for data in train_loader:
            # Break the loop if we exceed the max_iterations within the epoch
            if iteration >= max_iterations:
                break

            # Prepare input data and labels
            message = data["Message"]
            label = torch.tensor(data["Spam/Ham"], dtype=torch.long, device=device)

            # Perform model training step
            optimizer.zero_grad()
            prediction = model.predict_spam(message)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

            # Log and evaluate the model performance periodically
            if iteration % eval_interval == 0:
                test_accuracy = evaluate(encoder.model, test_1_loader)
                print(
                    f"Iteration: {iteration}, Loss: {loss.item()}, Test Accuracy: {test_accuracy}%"
                )
            elif iteration % log_interval == 0:
                print(f"Iteration: {iteration}, Loss: {loss.item()}")

            iteration += 1


def evaluate(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            message = data["Message"]
            labels = torch.tensor(data["Spam/Ham"], dtype=torch.long, device=device)
            outputs = encoder.predict_spam(message)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Set the model back to training mode
    accuracy = 100.0 * correct / total
    return accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train RoBERTa model for spam classification."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00006,
        help="Learning rate for optimizer.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("using learning rate", args.learning_rate)
    encoder = RoBERTaEncoder(device=device)
    optimizer = Adam(encoder.model.parameters(), lr=args.learning_rate)

    # Assuming SpamDataset has a method to return all data in a suitable format
    full_spam_dataset = SpamDataset(
        "enron_spam_data.csv", label="spam", sort_by_date=True
    )
    full_ham_dataset = SpamDataset(
        "enron_spam_data.csv", label="ham", sort_by_date=True
    )

    spam_dataset_splits = split_dataset(full_spam_dataset, 5)
    ham_dataset_splits = split_dataset(full_ham_dataset, 5)

    train_set_1 = combine_datasets(spam_dataset_splits[:2] + ham_dataset_splits[:2])
    train_set_2 = combine_datasets(spam_dataset_splits[:3] + ham_dataset_splits[:3])
    train_set_3 = combine_datasets(spam_dataset_splits[:4] + ham_dataset_splits[:4])

    test_set_1 = combine_datasets([spam_dataset_splits[2], ham_dataset_splits[2]])
    test_set_2 = combine_datasets([spam_dataset_splits[3], ham_dataset_splits[3]])
    test_set_3 = combine_datasets([spam_dataset_splits[4], ham_dataset_splits[4]])

    train_loader_1 = DataLoader(
        train_set_1,
        batch_size=128,
        shuffle=True,
        collate_fn=full_spam_dataset.spam_collate_fn,
    )

    train_loader_2 = DataLoader(
        train_set_2,
        batch_size=128,
        shuffle=True,
        collate_fn=full_spam_dataset.spam_collate_fn,
    )

    train_loader_3 = DataLoader(
        train_set_3,
        batch_size=128,
        shuffle=True,
        collate_fn=full_spam_dataset.spam_collate_fn,
    )

    test_1_loader = DataLoader(
        test_set_1,
        batch_size=128,
        shuffle=True,
        collate_fn=full_spam_dataset.spam_collate_fn,
    )
    test_2_loader = DataLoader(
        test_set_2,
        batch_size=128,
        shuffle=True,
        collate_fn=full_spam_dataset.spam_collate_fn,
    )
    test_3_loader = DataLoader(
        test_set_3,
        batch_size=128,
        shuffle=True,
        collate_fn=full_spam_dataset.spam_collate_fn,
    )

  
    loss_function = CrossEntropyLoss()
    print('------- Training self train Model (testing on temporal chunk 3/5) -------')
    train_model(
        encoder,
        train_loader=train_loader_1,
        test_loader=test_1_loader,
        device=device,
        optimizer=optimizer,
        max_iterations=100,
        log_interval=10,
        eval_interval=20,
    )
    
    print('------- Evaluating model on temporal chunk 3/5) -------')
    test_1_accuracy = evaluate(encoder.model, test_1_loader)
    print('Accuracy:', test_1_accuracy)
    print('------- Updating Oracal Model round  1 (testing on temporal chunk 3/5) -------')
    self_train(
        encoder,
        train_loader=test_1_loader,
        test_loader=test_1_loader,
        device=device,
        optimizer=optimizer,
        max_iterations=50,
        log_interval=5,
        eval_interval=10,
    )
    print('------- Evaluating model on temporal chunk 4/5) -------')
    test_2_accuracy = evaluate(encoder.model, test_2_loader)
    print('Accuracy:', test_2_accuracy)
    print('------- Updating Oracal Model round 2 (testing on temporal chunk 3/5) -------')
    self_train(
        encoder,
        train_loader=test_2_loader,
        test_loader=test_2_loader,
        device=device,
        optimizer=optimizer,
        max_iterations=50,
        log_interval=5,
        eval_interval=10,
    )

    print('------- Evaluating model on temporal chunk 5/5) -------')
    test_3_accuracy = evaluate(encoder.model, test_3_loader)
    print('Accuracy:', test_3_accuracy)