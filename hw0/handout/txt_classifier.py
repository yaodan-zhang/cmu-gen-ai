# From: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

import torch
import pandas as pd
import numpy as np
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
import time
import torchvision.transforms as T

import wandb
# log in to WanDB
wandb.login(key="d14b80fa525fa1077c2e99c01a4d1141ab0c37dc")


# Hyperparameters
EPOCHS = 5  # epoch
LR = 5  # learning rate
BATCH_SIZE = 8  # batch size for training
EMBED_DIM = 64 # embedding size in model
MAX_LEN = 1024 # maximum text input length

# Get cpu, gpu device for training.
# mps does not (yet) support nn.EmbeddingBag.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class CsvTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        text = self.data_frame.loc[idx, "article"]
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            text = self.transform(text)

        return text, label

class CorpusInfo():
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.oov_token = '<UNK>'
        self.pad_token = '<PAD>'
        
        def yield_tokens(data_iter):
            for text, _ in data_iter:
                yield tokenizer(text)
                
        self.vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=[self.oov_token, self.pad_token])
        self.vocab.set_default_index(self.vocab[self.oov_token])
        
        self.oov_idx = self.vocab[self.oov_token]
        self.pad_idx = self.vocab[self.pad_token]
        
        self.vocab_size = len(self.vocab)
        self.num_labels = len(set([label for (text, label) in dataset]))

class TextTransform(torch.Callable):
    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab

    '''def tokenize_and_numericalize(self, text):
        if isinstance(text, list):  # Ensure text is a string
            text = " ".join(text)  # Convert list back to string
        tokens = self.tokenizer(text)  # Tokenize the text
        return [self.vocab[token] for token in tokens]'''

    def tokenize_and_numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.vocab[token] for token in tokens]

    def __call__(self, text):
        return self.tokenize_and_numericalize(text)
    
class MaxLen(torch.Callable):
    def __init__(self, max_len):
        self.max_len = max_len
        
    def __call__(self, x):
        if len(x) > self.max_len:
            x = x[:self.max_len]
        return x
    
class PadSequence(torch.Callable):
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        def to_int_tensor(x):
            return torch.from_numpy(np.array(x, dtype=np.int64, copy=False))
        # Convert each sequence of tokens to a Tensor
        sequences = [to_int_tensor(x[0]) for x in batch]
        # Convert the full sequence of labels to a Tensor
        labels = to_int_tensor([x[1] for x in batch])
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.pad_idx)
        return sequences_padded, labels

def get_data():    
    train_data = CsvTextDataset(
        csv_file='./data/txt_train.csv',
        transform=None,
    )
    tokenizer = get_tokenizer("basic_english")
    corpus_info = CorpusInfo(train_data, tokenizer)

    transform_txt = T.Compose([
        TextTransform(corpus_info.tokenizer, corpus_info.vocab),
        MaxLen(MAX_LEN),
    ])

    train_data = CsvTextDataset(
        csv_file='./data/txt_train.csv',
        transform=transform_txt,
    )
    
    val_data = CsvTextDataset(
        csv_file='./data/txt_val.csv',
        transform=transform_txt,
    )
    test_data = CsvTextDataset(
        csv_file='./data/txt_test.csv',
        transform=transform_txt,
    )
    # Compute article lengths **before** applying transformations
    article_lengths = [len(l) for l, _ in train_data]
    wandb.log({"Article Lengths": wandb.Histogram(article_lengths)}, step=0)

    collate_batch = PadSequence(corpus_info.pad_idx)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    for X, y in train_dataloader:
        print(f"Shape of X [B, N]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return corpus_info, train_dataloader, val_dataloader, test_dataloader

'''class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)'''

class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, num_layers=1, bidirectional=False):
        super(LSTMTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Embedding layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Compute final output size
        lstm_output_size = hidden_dim * (2 if bidirectional else 1)
        
        self.fc = nn.Linear(lstm_output_size, num_class)  # Fully connected layer
        
    def forward(self, text):
        """
        text: Tensor of shape [batch_size, seq_length]
        """
        embedded = self.embedding(text)  # Convert indices to word embeddings
        lstm_out, _ = self.lstm(embedded)  # Pass through LSTM
        
        # Use the last time step's hidden state for classification
        last_hidden_state = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_dim]
        
        output = self.fc(last_hidden_state)  # Pass through fully connected layer
        return output


def train_one_epoch(dataloader, model, criterion, optimizer, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 5
    start_time = time.time()

    for idx, (text, label) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def main():
    wandb.init(project="text_classification_histogram", name="run-with-LSTM-Model")

    corpus_info, train_dataloader, val_dataloader, test_dataloader = get_data()

    '''model = TextClassificationModel(corpus_info.vocab_size, EMBED_DIM, corpus_info.num_labels).to(device)'''

    HIDDEN_DIM = 128  # Number of LSTM units per layer
    NUM_LAYERS = 2    # Stacked LSTM layers
    BIDIRECTIONAL = True  # Use bidirectional LSTM

    model = LSTMTextClassifier(
        vocab_size=corpus_info.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_class=corpus_info.num_labels,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    total_accu = None    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train_one_epoch(train_dataloader, model, criterion, optimizer, epoch)
        accu_val = evaluate(val_dataloader, model, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader, model, criterion)
    print("test accuracy {:8.3f}".format(accu_test))

    wandb.log({"Validation Accuracy": total_accu, "Test Accuracy": accu_test})  

if __name__ == '__main__':
    main()