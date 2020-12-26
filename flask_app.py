from flask import Flask, request

import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image

app = Flask(__name__)

app.config["DEBUG"] = True



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



def compute(external_image):
    model = ResNet()
    checkpoint = torch.load('/home/flutterpython/mysite/datasets/new_torch.zip', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image = Image.open(external_image)
    input = trans(image)
# ['paper', 'metal', 'plastic', 'glass']
    input = input.view(1, 3, 256, 256)
    output = model(input)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    return prediction



@app.route('/API', methods=['POST'])
def eco_sorter():
    image = request.files['file']
    return str(compute(image))
