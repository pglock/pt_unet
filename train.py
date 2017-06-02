import torch
from unet import UNET

def hellinger_distance(y_true, y_pred):
    dif = torch.sqrt(y_true) - torch.sqrt(y_pred)
    dif = torch.pow(dif, 2)
    dif = torch.sqrt(torch.sum(dif)) / 1.4142135623730951
    return dif

if __name__ == "__main__":
    net = UNET()
    criterion = hellinger_distance
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

