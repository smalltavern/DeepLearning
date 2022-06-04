"""
    1. 准备数据
    2. 构建模型
    3. 模型的训练
    4. 模型的评估
    5. 模型的保存
"""
import torch
from torchvision import transforms
import numpy as np
import torchvision
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

# 设置超参数
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

train_losses = []
train_counter = []
test_losses = []


def get_dataloader(train=True):
    assert isinstance(train, bool)

    # 加载训练数据
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    # 加载测试数据
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader


def get_parameter_number(model_analyse):
    #  打印模型参数量
    total_num = sum(p.numel() for p in model_analyse.parameters())
    trainable_num = sum(p.numel() for p in model_analyse.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)


def define_important_things(model):
    # 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=momentum)
    return criterion, optimizer


def train(epochs, model, optimizer, criterion, train_loader):
    # 训练模型
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # 迭代十次打印一下，并且保存模型和优化的参数
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(model.state_dict(), './model.pth')
                torch.save(optimizer.state_dict(), './optimizer.pth')


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def print_orignal_picture(loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def pridect_picture(loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = model(example_data)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def show_line_chart():
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


class NetMNIST(nn.Module):
    def __init__(self):
        super(NetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.sigmoid(x) 加sigmoid后loss降不下去
        return x


if __name__ == "__main__":
    train_loader, test_loader = get_dataloader(train=True)
    # print('len(train_loader): ', len(train_loader))
    # print('len(test_loader):  ', len(test_loader))
    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print(example_targets)
    # print(example_data.shape)  # [64, 1, 28, 28]
    # fig = plt.figure()
    # print_orignal_picture(loader=train_loader) # 展示部分图片和标签

    # 数据的训练,测试,预测
    model = NetMNIST()
    criterion, optimizer = define_important_things(model=model)
    print(get_parameter_number(model))
    train(epochs=6, model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader)
    test(model=model, test_loader=test_loader)
    show_line_chart()
    pridect_picture()

