import torch
import torch.nn as nn
from torch.autograd import Variable
from unet import UNET
from data import create_patches
import numpy as np
from itertools import zip_longest

num_epochs = 12
batch_size = 32

def hellinger_distance(y_true, y_pred):
    dif = torch.sqrt(y_true) - torch.sqrt(y_pred)
    dif = torch.pow(dif, 2)
    dif = torch.sqrt(torch.sum(dif)) / 1.4142135623730951
    return dif

def train():
    cuda = torch.cuda.is_available()
    net = UNET()
    if cuda:
        net = net.cuda()
    criterion = hellinger_distance
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    print("preparing training data ...")
    train_set = create_patches("./imgs", "./masks")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("done ...")

    test_set = create_patches("./test_imgs", "./test_masks", n_patches=10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    for epoch in range(num_epochs):
        net.train()
        for i, (images, masks) in enumerate(train_loader):
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

        net.eval()
        for images, masks in test_loader:
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            outputs = net(images)
            val_loss = criterion(outputs, masks)

        print("Epoch {}, Loss: {}, Validation Loss: {}".format(epoch+1, loss.data[0], val_loss.data[0]))

    return net

def test(model):
    model.eval()



if __name__ == "__main__":
    train()
