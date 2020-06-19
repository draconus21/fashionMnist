import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model.simple as simple
import utils.helpers as helpers

from torch.utils.tensorboard import SummaryWriter


def train(trainset, classes, writer, criterion, optimizer, net):
    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=2)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    img_grid = torchvision.utils.make_grid(images)
#

    helpers.matplotlib_imshow(img_grid, ax=None, one_channel=True)

    writer.add_image('four_fashion_mnist_images', img_grid)
    writer.add_graph(net, images)
    writer.close()

    # select random images and their target indices
    images, labels = helpers.select_n_random(trainset.data, trainset.targets)
    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 28*28)
    writer.add_embedding(features, metadata=class_labels,
                         label_img=images.unsqueeze(1))
    writer.close()

    running_loss = 0.0
    nepochs = 1
    for epoch in range(nepochs):
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%1000 == 999:
                # ...log the running loss
                writer.add_scalar('training loss',
                                  running_loss/1000,
                                  epoch * len(trainloader) + i)

                # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
                writer.add_figure('predictions vs. actuals',
                                  helpers.plot_classes_preds(net, inputs, labels, classes),
                                                     global_step=epoch*len(trainloader) + i)
                running_loss = 0.0

    print('Fininshed Training')

def add_pr_curve_tensorboard(class_index, test_probs, test_preds,
                             classes, writer, global_step=0):
    '''
    takes in a "class_index" from 0 to 9 and plots the corresponding preciision-recall curve
    '''

    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

def test(testset, classes, writer, net):
    class_probs = []
    class_preds = []

    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=2)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds, classes, writer)



def run():
    # transforms
    transform = transforms.Compose(
                  [transforms.ToTensor(),
                   transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
                                               download=True,
                                               train=True,
                                               transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
                                               download=True,
                                               train=False,
                                               transform=transform)

    # constants for classes
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
              'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
              'Ankle Boot']


    net = simple.Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    train(writer=writer, trainset=trainset, classes=classes,
          criterion=criterion, optimizer=optimizer,
          net=net)

    test(writer=writer, testset=testset, classes=classes,
          net=net)

if __name__ == '__main__':
    run()
