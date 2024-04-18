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
        for data in tqdm(data_loader):
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
    #encoder.save_weights('baseline_spam_roberta.pt')


    # Assuming SpamDataset has a method to return all data in a suitable format
    full_spam_dataset = SpamDataset(
        "enron_spam_data.csv", label="spam", sort_by_date=True
    )
    full_ham_dataset = SpamDataset(
        "enron_spam_data.csv", label="ham", sort_by_date=True
    )

    spam_dataset_splits = split_dataset(full_spam_dataset, 5)
    ham_dataset_splits = split_dataset(full_ham_dataset, 5)

    train_set = combine_datasets(spam_dataset_splits[:-1] + ham_dataset_splits[:-1])
    test_set_1 = combine_datasets([spam_dataset_splits[2], ham_dataset_splits[2]])
    test_set_2 = combine_datasets([spam_dataset_splits[3], ham_dataset_splits[3]])
    test_set_3 = combine_datasets([spam_dataset_splits[4], ham_dataset_splits[4]])

    train_loader = DataLoader(
        train_set,
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
    print('------- Training Baseline Model (testing on temporal chunk 3/5) -------')
    test_1_accuracy = evaluate(encoder.model, test_1_loader)
    print('Accuracy:', test_1_accuracy)

    train_model(
        encoder,
        train_loader=train_loader,
        test_loader=test_1_loader,
        device=device,
        optimizer=optimizer,
        max_iterations=60,
        log_interval=10,
        eval_interval=20,
    )
    encoder.load_weights('baseline_spam_roberta.pt')
    test_3_accuracy = evaluate(encoder.model, test_1_loader)


    print('------- Evaluating model on temporal chunk 3/5) -------')
    test_1_accuracy = evaluate(encoder.model, test_1_loader)
    print('Accuracy:', test_1_accuracy)

    print('------- Evaluating model on temporal chunk 4/5) -------')
    test_2_accuracy = evaluate(encoder.model, test_2_loader)
    print('Accuracy:', test_2_accuracy)

    print('------- Evaluating model on temporal chunk 5/5) -------')
    test_3_accuracy = evaluate(encoder.model, test_3_loader)
    print('Accuracy:', test_3_accuracy)

    encoder.load_weights('baseline_spam_roberta.pt')
    test_3_accuracy = evaluate(encoder.model, test_3_loader)
    print('initial acc sanitycheck:', test_3_accuracy)

