import torch
import torch.nn as nn
from torch.autograd import Variable
from unet import UNET
from data import create_patches
import numpy as np
from itertools import zip_longest

num_epochs = 12
batch_size = 32


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    group = []
    for element in iterable:
        group.append(element)
        if len(group) == n:
            group = torch.LongTensor(group)
            yield group
            group = []
    group = torch.LongTensor(group)
    yield group

def hellinger_distance(y_true, y_pred):
    dif = torch.sqrt(y_true) - torch.sqrt(y_pred)
    dif = torch.pow(dif, 2)
    dif = torch.sqrt(torch.sum(dif)) / 1.4142135623730951
    return dif

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    net = UNET()
    if cuda:
        net = net.cuda()
    criterion = hellinger_distance
    optimizer = torch.optim.Adamax(net.parameters(), lr=1e-5)

    print("preparing training data ...")
    train_set = create_patches("./imgs", "./masks")
    print("done ...")
    index = np.array(range(train_set[0].size()[0]), dtype=np.int)
    np.random.shuffle(index)
    print(len(index))
    index = torch.from_numpy(index)
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        batch = grouper(index, batch_size)
        for i, ids in enumerate(batch):
            images = train_set[0][ids]
            masks = train_set[1][ids]
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        print("Epoch {}, Loss: {}".format(epoch+1, loss.data[0]))
