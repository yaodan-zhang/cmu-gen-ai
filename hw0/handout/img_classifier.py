import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse

import wandb
# log in to WanDB
wandb.login(key="d14b80fa525fa1077c2e99c01a4d1141ab0c37dc")

# Map label indices to class names
label_names = {0: "parrot", 1: "narwhal", 2: "axolotl"}
num_labels = 3
img_size = (256,256)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()

        # First Convolution (k=4) -> Output: (B, 128, 64, 64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=4, padding=0)
        self.ln1 = nn.LayerNorm([128, 64, 64])  # Layer Norm on channels

        # Second Convolution (k=7) -> Output: (B, 128, 64, 64)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.ln2 = nn.LayerNorm([128, 64, 64])  # Layer Norm on channels

        # First Linear Layer -> Expands channels to 256
        self.fc1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)  # (B, 256, 64, 64)
        self.gelu = nn.GELU()  # GELU Activation

        # Second Linear Layer -> Reduces channels back to 128
        self.fc2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)  # (B, 128, 64, 64)

        # 2D Average Pooling (k=2) -> Output: (B, 128, 32, 32)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Flatten the feature maps for classification
        self.flatten = nn.Flatten()  # (B, 128 * 32 * 32)

        # Final Fully Connected Layer -> (B, 3) for classification
        self.fc3 = nn.Linear(128 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)

        x = self.conv2(x)
        x = self.ln2(x)

        x = self.fc1(x)
        x = self.gelu(x)

        x = self.fc2(x)

        x = self.avg_pool(x)
        x = self.flatten(x)

        x = self.fc3(x)  # Final classification layer
        return x

# Define the inverse normalization function
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)  # Reshape for broadcasting
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean  # Reverse the normalization

def log_images(dataloader, model, dataset_name):
    model.eval()  # Ensure model is in evaluation mode
    transform_to_pil = T.ToPILImage()  # Convert tensor to image format

    images, labels = next(iter(dataloader))  # Get the first batch
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        predictions = model(images).argmax(dim=1)  # Get predicted labels

    # Denormalize images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denorm_images = denormalize(images.cpu(), mean, std).clamp(0, 1)  # Clamp values between 0 and 1

    wandb_images = []
    for i in range(len(denorm_images)):
        image = transform_to_pil(denorm_images[i])  # Convert to PIL image
        pred_label = label_names[predictions[i].item()]
        true_label = label_names[labels[i].item()]
        caption = f"{pred_label} / {true_label}"

        wandb_images.append(wandb.Image(image, caption=caption))

    wandb.log({f"{dataset_name} Images": wandb_images})


class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        image = Image.open(img_name).convert("RGB")  # Assuming RGB images
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(batch_size):
    # Original transform
    transform_img = T.Compose([
        T.ToTensor(), 
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        T.CenterCrop(img_size),  # Center crop to 256x256
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize each color dimension
    ])
    
    # Grey scale transform
    '''transform_img = T.Compose([
        T.Grayscale(num_output_channels=1),  # Convert to grayscale
        T.ToTensor(),
        T.Resize(min(img_size[0], img_size[1]), antialias=True),
        T.CenterCrop(img_size),
        T.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
    ])'''
    
    # Tiny size transform
    '''transform_img = T.Compose([
        T.ToTensor(),
        T.Resize((28, 28), antialias=True),  # Resize to 28x28
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Keep normalization
    ])'''

    train_data = CsvImageDataset(
        csv_file='./data/img_train.csv',
        transform=transform_img,
    )

    test_data = CsvImageDataset(
        csv_file='./data/img_test.csv',
        transform=transform_img,
    )

    val_data = CsvImageDataset(
        csv_file='./data/img_val.csv',
        transform=transform_img)  # Load validation data

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)  # Create validation dataloader
    
    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader,  val_dataloader, test_dataloader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Original model
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size[0] * img_size[1] * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

        # Greyscale model
        '''self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size[0] * img_size[1] * 1, 512),  # 1 channel instead of 3
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )'''

        # Tiny size model
        '''self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28 * 3, 512),  # Use 3 channels (RGB)
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )'''

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_one_epoch(dataloader, model, loss_fn, optimizer, t, total_examples):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss = loss.item() / batch_size
        total_examples += len(X)  # Track the number of examples seen so far

        # Log batch loss to WandB
        wandb.log({
            "batch_loss": avg_loss,
            "examples_seen": total_examples
        })
        # current = (batch + 1) * dataloader.batch_size
        if batch % 10 == 0:
            print(f"Train loss = {avg_loss:>7f}  [{total_examples:>5d}/{size:>5d}]")

    return total_examples
        
def evaluate(dataloader, dataname, model, loss_fn):
    size = len(dataloader.dataset)
    #num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss /= size
    correct /= size
    accuracy = 100 * correct

    print(f"{dataname} accuracy = {accuracy:>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")
    return accuracy, avg_loss  # Return values for logging in WandB


def main(n_epochs, batch_size, learning_rate, optimizer_name):
    print(f"Using {device} device")

    # Different Optimizer
    wandb.init(project="image-classifier", name="Complex Network", config={
        "epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name
    })

    # Initialize Weights & Biases
    '''wandb.init(project="image-classifier", name="neural-the-narwhal", config={
        "epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })'''

    # Greyscale run
    '''wandb.init(project="image-classifier", name="grayscale-experiment", config={
    "epochs": n_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "image_mode": "grayscale"
    })'''

    # Tiny size run
    '''wandb.init(project="image-classifier", name="tiny-images-28x28", config={
    "epochs": n_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "image_size": (28, 28)
    })'''

    train_dataloader, val_dataloader, test_dataloader = get_data(batch_size)
    
    #model = NeuralNetwork().to(device)
    model = CustomCNN(num_classes=3).to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()

    # ✅ Select the optimizer
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    total_examples = 0  # Track total examples seen

    for t in range(n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")

        total_examples = train_one_epoch(train_dataloader, model, loss_fn, optimizer, t, total_examples)
        train_acc, train_loss = evaluate(train_dataloader, "Train", model, loss_fn)
        val_acc, val_loss = evaluate(val_dataloader, "Validation", model, loss_fn)  # Compute validation metrics
        test_acc, test_loss = evaluate(test_dataloader, "Test", model, loss_fn)

        # Log train & validation metrics to WandB
        wandb.log({
            "epoch": t + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

        # On the last epoch, log images with predictions
        if t == n_epochs - 1:
            log_images(train_dataloader, model, "Train")
            log_images(val_dataloader, model, "Validation")
            log_images(test_dataloader, model, "Test")

    print("Done!")

    torch.save(model.state_dict(), f"model_{optimizer_name}.pth")
    print(f"Saved PyTorch Model State to model_{optimizer_name}.pth")

    model = CustomCNN().to(device)
    model.load_state_dict(torch.load(f"model_{optimizer_name}.pth"))

    # Finish WandB run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--n_epochs', default=5, help='The number of training epochs', type=int)
    parser.add_argument('--batch_size', default=8, help='The batch size', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for the optimizer', type=float)
    parser.add_argument('--optimizer_name', default="SGD", help='The optimizer type', type=str)

    args = parser.parse_args()
        
    main(args.n_epochs, args.batch_size, args.learning_rate, args.optimizer_name)
    
# Original MLP Model → 100.9M parameters (fully connected, inefficient).
# New CustomCNN Model → 1.3M parameters (convolution-based, efficient).

'''Summary:
MLPs are good for structured/tabular data, but not ideal for image classification.
CNNs learns hierarchical patterns (edges → textures → object parts → objects) - are the best choice for image classification!
'''