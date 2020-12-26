import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image


#
# def predict_image(img, model):
#     # Convert to a batch of 1
#     xb = to_device(img.unsqueeze(0), device)
#     # Get predictions from model
#     yb = model(xb)
#     # Pick index with highest probability
#     prob, preds = torch.max(yb, dim=1)
#     # Retrieve the class label
#     print(dataset.classes)
#     return dataset.classes[preds[0].item()]


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


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


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 4)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


def compute(ext_image):
    model = ResNet()
    checkpoint = torch.load('datasets/new_torch.zip', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image = Image.open(ext_image)
    input = trans(image)
    input = input.view(1, 3, 256, 256)
    output = model(input)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    print(prediction)


#
# def to_device(data, device):
#     """Move tensor(s) to chosen device"""
#     if isinstance(data, (list, tuple)):
#         return [to_device(x, device) for x in data]
#     return data.to(device, non_blocking=True)
#
#
#
# def get_default_device():
#     """Pick GPU if available, else CPU"""
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')
#
# def predict_external_image(image_name):
#     image = Image.open(image_name)
#
#     example_image = transformations(image)
#     plt.imshow(example_image.permute(1, 2, 0))
#     print("The image resembles", predict_image(example_image, model))

#
# transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#
# data_dir = 'datasets/garbage classification/Garbage classification'
#
# dataset = ImageFolder(data_dir, transform=transformations)
# device = get_default_device()
#
# model = ResNet()
# model.load_state_dict(torch.load('datasets/new_torch.zip'))
#
# if torch.cuda.is_available():
#     model.cuda()
# # ['glass', 'metal', 'paper', 'plastic']
# predict_external_image('datasets/garbage classification/Garbage classification/plastic/plastic331.jpg')

compute('datasets/garbage classification/Garbage classification/plastic/plastic90.jpg')
