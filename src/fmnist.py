import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from model import Net

def matplotlib_imshow(img, ax, one_channel=False):
    if ax is None:
        ax = plt.figure().gca()
    if one_channel:
        img = img.mean(dim=0)
    img = img/2 + 0.5 # unnormalize
    npimg = img.numpy()

    if one_channel:
        ax.imshow(npimg, cmap='Greys')
    else:
        ax.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained network and a list of
    images'
    '''
    output = net(images)

    # convert optput probabilities to a predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images, and lables from a
    batch, that shows the network's top prediction along with its probability, alongside the
    actual label, coloring this infomration based on wheterh the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''

    print(images.shape)
    preds, probs = images_to_probs(net, images)
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 48), squeeze=False)
    for idx in np.arange(4):
        matplotlib_imshow(images[idx], ax=ax[0, idx], one_channel=True)
        ax[0, idx].set_title('{0}, {1:.1f}%\nlabel: {2}'.format(
            classes[preds[idx]],
            probs[idx]*100.0,
            classes[labels[idx]]),
            color=('green' if preds[idx]==labels[idx].item() else 'red'))
    return fig

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints  and their corresponding labels from a dataset
    '''

    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

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

    matplotlib_imshow(img_grid, ax=None, one_channel=True)

    writer.add_image('four_fashion_mnist_images', img_grid)
    writer.add_graph(net, images)
    writer.close()

    # select random images and their target indices
    images, labels = select_n_random(trainset.data, trainset.targets)
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
                                  plot_classes_preds(net, inputs, labels, classes),
                                                     global_step=epoch*len(trainloader) + i)
                running_loss = 0.0

    print('Fininshed Training')




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

    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=2)

    # constants for classes
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
              'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
              'Ankle Boot']


    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    train(writer=writer, trainset=trainset, classes=classes,
          criterion=criterion, optimizer=optimizer,
          net=net)


if __name__ == '__main__':
    run()
