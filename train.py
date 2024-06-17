## Training Phase

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
num_epochs = 90
batch_size = 128
initial_learning_rate = 0.01

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total_batches = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            total_batches += 1

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= total_batches
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

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

optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
PATH = './improved_cnn.pth'

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

best_val_acc = 0.0
for epoch in range(num_epochs):
    if epoch == 20:
        adjust_learning_rate(optimizer, 0.005)
    elif epoch == 29:
        adjust_learning_rate(optimizer, 0.001)
    elif epoch == 35:
        adjust_learning_rate(optimizer, 0.0005)
    elif epoch == 50:
        adjust_learning_rate(optimizer, 0.0001)
    elif epoch == 60:
        adjust_learning_rate(optimizer, 0.00001)

    model.train()
    running_loss = 0.0
    n_total_steps = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % (n_total_steps // 5) == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {running_loss / (i + 1):.4f}')
            running_loss = 0.0

    val_loss, val_acc = evaluate(model, test_loader, device)

    scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

torch.save(model.state_dict(), PATH)

print('Finished Training')

model.load_state_dict(torch.load(PATH))

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for label, pred in zip(labels, predicted):
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.2f} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.2f} %')


# Epoch [90/90], Validation Loss: 0.3919, Validation Accuracy: 89.41%
# Finished Training
# Accuracy of the network: 89.41 %
# Accuracy of plane: 91.40 %
# Accuracy of car: 94.90 %
# Accuracy of bird: 85.20 %
# Accuracy of cat: 79.00 %
# Accuracy of deer: 89.50 %
# Accuracy of dog: 82.30 %
# Accuracy of frog: 92.00 %
# Accuracy of horse: 91.60 %
# Accuracy of ship: 94.10 %
# Accuracy of truck: 94.10 %
