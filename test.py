## Testing Phase

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ImprovedConvNet(nn.Module):
    def __init__(self):
        super(ImprovedConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.batch_norm_fc1 = nn.BatchNorm1d(1024)
        self.batch_norm_fc2 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.batch_norm_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = ImprovedConvNet().to(device)
model.load_state_dict(torch.load('./drive/MyDrive/Demo-cifar/improved_cnn.pth', map_location=device))
model.eval()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

def imshow(img):
    npimg = img.numpy()
    plt.figure(figsize=(0.8, 0.8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def predict_single_image(model, test_dataset, device):
    model.eval()
    idx = np.random.randint(len(test_dataset))
    image, label = test_dataset[idx]
    imshow(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]

    print(f'Predicted class: {predicted_class}')
    print(f'Actual class: {classes[label]}')

predict_single_image(model, test_dataset, device)
