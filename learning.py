import os
import torch
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from PIL import Image

# Path of data set
data_dir = 'datasets/garbage classification/Garbage classification'

# Listing classes & printing it in the terminal (optional)
classes = os.listdir(data_dir)
print(classes)

# Transformations Resizes all the input image to the given size and transforms it to arrays to operate on.
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Defining data set variable which loads all datasets.
dataset = ImageFolder(data_dir, transform=transformations)


# Function to test any sample in the samples (optional)
def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: " + str(label) + ")")

    plt.imshow(img.permute(1, 2, 0))


# Testing out showing sample (optional)
img, label = dataset[12]
show_sample(img, label)

# Random number
random_seed = 42
# Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration
batch_size = 8
# sets the random seed from pytorch random number generators
torch.manual_seed(random_seed)
# Splitting data set into 3 data sets [Training data set] , [Validation data set] & [Testing data set]
train_ds, val_ds, test_ds = random_split(dataset, [1900, 1004, 1070])
len(train_ds), len(val_ds), len(test_ds)

# Combining data sets using data loader
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)


# This is a helper function to visualize batches (Optional)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


show_batch(train_dl)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Model base , Explanation :
"""
Model based machine learning is a machine learning algorithm.
that uses heuristics, simple summary statistics, randomness,
or machine learning to create predictions for a dataset. 
You can use these predictions to measure the baseline's performance (e.g., accuracy)
A machine learning algorithm tries to learn a function that models the relationship between the input 
(feature) data and the target variable (or label). 
When you test it, you will typically measure performance in one way or another. 
For example, your algorithm may be 75% accurate. But what does this mean? 
You can infer this meaning by comparing with a baseline's performance.
"""


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))


# We'll be using ResNet50 for classifying images , What is ResNet?:

"""
ResNet-50 is a convolutional neural network that is 50 layers deep. 
You can load a pretrained version of the network trained on more than a million images from the 
ImageNet database . The pretrained network can classify images into 1000 object categories, 
such as keyboard, mouse, pencil, and many animals.
"""


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


model = ResNet()

"""
## CHECK WHAT IS CPU VS GPU.
GPU is faster than CPU , so we will train models using the GPU speed.
During the tests, some hyper parameters were adjusted and the performance values
were compared between CPU and GPU. 
It has been observed that the GPU runs faster than the CPU in all tests performed. 
In some cases, GPU is 4-5 times faster than CPU, according to the tests performed on GPU server 
and CPU server.
"""


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)

# This is the function for fitting the model.

"""
Model fitting is a measure of how well a machine learning model generalizes
to similar data to that on which it was trained. 
A model that is well-fitted produces more accurate outcomes. 
A model that is overfitted matches the data too closely. 
A model that is underfitted doesn’t match closely enough.
Model fitting is the essence of machine learning. If your model doesn’t fit your data correctly, 
the outcomes it produces will not be accurate enough to be useful for practical decision-making. 
"""


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Let's start training the model:

model = to_device(ResNet(), device)
evaluate(model, val_dl)

"""
An epoch is a term used in machine learning and indicates the number of passes
of the entire training dataset the machine learning algorithm has completed. 
If the batch size is the whole training dataset then the number of epochs is the number of iterations.
"""
num_epochs = 8
# Adam Algorithmc
opt_func = torch.optim.Adam
"""
Logistic regression is another technique borrowed by machine learning from the field of statistics.
It is the go-to method for binary classification problems (problems with two class values)

1 / (1 + e^-value)
"""
lr = 5.5e-5

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
torch.save(model.state_dict(), 'datasets/new_torch.zip')


# Plotting accuracies (optional)
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


plot_accuracies(history)


# Predictions of model
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


# Let us see the model's predictions on the test dataset:

img, label = test_ds[17]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
loaded_model = model


# Let's now test with external images.
def predict_external_image(image_name):
    image = Image.open(image_name)

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, loaded_model) + ".")


predict_external_image('datasets/garbage classification/Garbage classification/metal/metal16.jpg')
