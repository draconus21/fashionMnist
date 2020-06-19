import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

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

