import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from flask import Flask, request, render_template
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['hyundai', 'lexus', 'mazda', 'mercedes', 'opel', 'skoda', 'toyota', 'volkswagen']


# Load your trained model
class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, stride=2)  # Pooling layer with kernel size of 2
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


app = Flask(__name__)

model = ConvNeuralNet()
model.load_state_dict(torch.load('model5-many_transf-50-54.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    # Get the uploaded image
    file = request.files['file']
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = classes[predicted.item()]

    return render_template('result.html', prediction=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)
