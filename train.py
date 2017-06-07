from data import create_patches
from torch.autograd import Variable
from unet import UNET
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

num_epochs = 12
batch_size = 32

torch.manual_seed(42)

def hellinger_distance(y_pred, y_true, size_average=True):
    n = y_pred.size(0)
    dif = torch.sqrt(y_true) - torch.sqrt(y_pred)
    dif = torch.pow(dif, 2)
    dif = torch.sqrt(torch.sum(dif)) / 1.4142135623730951
    if size_average:
        dif = dif / n
    return dif

class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

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
        train_loss = Average()
        net.train()
        for i, (images, masks) in enumerate(train_loader):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks, size_average=False)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.data[0], images.size(0))

        val_loss = Average()
        net.eval()
        for images, masks in test_loader:
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            outputs = net(images)
            vloss = criterion(outputs, masks, size_average=False)
            val_loss.update(vloss.data[0], images.size(0))

        print("Epoch {}, Loss: {}, Validation Loss: {}".format(epoch+1, train_loss.avg, val_loss.avg))

    return net

def test(model):
    model.eval()



if __name__ == "__main__":
    train()
