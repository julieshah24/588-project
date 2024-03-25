import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from text_encoders import RoBERTaEncoder
from datasets import SpamDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse

'''

def train_and_evaluate(
    encoder,
    dataset,
    train_size=0.8,
    batch_size=32,
    iterations=1000,
    lr=1e-5,
    eval_every_n_itr=200,
):

    train_indices, test_indices = train_test_split(
        range(len(dataset)), train_size=train_size, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = BCEWithLogitsLoss()
    optimizer = Adam(encoder.model.parameters(), lr=lr)

    for itr in range(iterations):

        encoder.model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Iteration {itr+1}/{itr}"):
            optimizer.zero_grad()

            # this may be not the best way
            texts = [sample["Subject"] + " " + sample["Message"] for sample in batch]
            labels = torch.tensor(
                [sample["Spam/Ham"] for sample in batch], dtype=torch.float
            ).to(encoder.device)

            encoded_texts = torch.stack([encoder.encode(text) for text in texts])
            logits = encoder.predict(encoded_texts)
            loss = criterion(logits.squeeze(), labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print(
            f"Iteration {itr+1}, Train Loss: {total_train_loss / len(train_dataloader)}"
        )

        if (itr + 1) % eval_every_n_itr == 0:
            print('#####################')
            print('### training loss ###')
            evaluate(encoder, train_dataloader, criterion)
            print('### testing loss ###')
            evaluate(encoder, test_dataloader, criterion)
            print('#####################')

def evaluate(encoder, dataloader, criterion):
    encoder.model.eval()
    total_eval_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = [sample["Subject"] + " " + sample["Message"] for sample in batch]
            labels = torch.tensor(
                [sample["Spam/Ham"] for sample in batch], dtype=torch.float
            ).to(encoder.device)

            encoded_texts = torch.stack([encoder.encode(text) for text in texts])
            logits = encoder.predict(encoded_texts)
            loss = criterion(logits.squeeze(), labels)

            total_eval_loss += loss.item()
            preds = torch.round(torch.sigmoid(logits.squeeze()))
            correct_predictions += (preds == labels).sum().item()

    avg_loss = total_eval_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    print(f"Evaluation Loss: {avg_loss}, Accuracy: {accuracy}")


device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = RoBERTaEncoder(device=device, size="base")
dataset = SpamDataset(csv_file="enron_spam_data.csv")
train_and_evaluate(encoder, dataset)
'''
from sklearn.model_selection import train_test_split

def evaluate(model,data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            message = data['Message']
            labels = torch.tensor(data['Spam/Ham'], dtype=torch.long, device=device)
            outputs = encoder.predict_spam(message)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Set the model back to training mode
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train RoBERTa model for spam classification.")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="Learning rate for optimizer.")
    args = parser.parse_args()
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using learning rate', args.learning_rate)
    encoder = RoBERTaEncoder(device=device)
    for name, param in encoder.model.named_parameters():
        print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")

    optimizer = Adam(encoder.model.parameters(), lr=args.learning_rate)#,weight_decay=1e-4)
    loss_function = CrossEntropyLoss()

    # Assuming SpamDataset has a method to return all data in a suitable format
    full_dataset = SpamDataset('enron_spam_data.csv', sort_by_date=True)
    train_data, test_data = train_test_split(full_dataset, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=full_dataset.spam_collate_fn)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=full_dataset.spam_collate_fn)
    iteration = 0
    max_iterations = 10000
    while iteration < max_iterations:
        for data in train_loader:
            message = data['Message']
            label = torch.tensor(data['Spam/Ham'], dtype=torch.long, device=device)
            optimizer.zero_grad()
            prediction = encoder.predict_spam(message)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            if iteration % 100 == 0:
                # Evaluate on test set
                test_accuracy = evaluate(encoder.model,test_loader)
                #train_accuracy = evaluate(encoder.model,train_loader)
                print(f"Iteration: {iteration}, Loss: {loss.item()}, Test Accuracy: {test_accuracy}%")
            elif iteration % 10 == 0:
                print(f"Iteration: {iteration}, Loss: {loss.item()}")
            iteration += 1

