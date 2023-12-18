# import dependencies
import torch.nn as nn


# Simple implementation of a convolutional neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # shape: (100, 100, 1) -> shape (3)

        # shape: (100, 100, 1); Params: 20
        self.cl1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1)
        self.at1 = nn.LeakyReLU()

        # shape: (50, 50, 2); Params: 76
        self.cl2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.at2 = nn.LeakyReLU()
        # shape: (25, 25, 4); Params: 296
        self.cl3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.at3 = nn.LeakyReLU()

        # shape: (13, 13, 8)
        self.flatten = nn.Flatten(0, 2)

        # shape: (1352); Params: 1385800
        self.fcn1 = nn.Linear(in_features=1352, out_features=1024)
        self.at4 = nn.LeakyReLU()

        # shape: (1024); Params: 263168
        self.fcn2 = nn.Linear(in_features=1024, out_features=256)
        self.at5 = nn.LeakyReLU()

        # shape: (256); Params: 16640
        self.fcn3 = nn.Linear(in_features=256, out_features=64)
        self.at6 = nn.LeakyReLU()

        # shape: (64); Params: 1088
        self.fcn4 = nn.Linear(in_features=64, out_features=16)
        self.at7 = nn.LeakyReLU()

        # shape: (16); Params: 80
        self.fcn5 = nn.Linear(in_features=16, out_features=4)
        self.at8 = nn.LeakyReLU()

        # Params: 16
        self.fcn6 = nn.Linear(in_features=4, out_features=3)

        # total (trainable) params: 1,667,184

    def forward(self, x):
        x = self.cl1(x)
        x = self.at1(x)
        x = self.cl2(x)
        x = self.at2(x)
        x = self.cl3(x)
        x = self.at3(x)
        x = self.flatten(x)
        x = self.fcn1(x)
        x = self.at4(x)
        x = self.fcn2(x)
        x = self.at5(x)
        x = self.fcn3(x)
        x = self.at6(x)
        x = self.fcn4(x)
        x = self.at7(x)
        x = self.fcn5(x)
        x = self.at8(x)
        x = self.fcn6(x)

        return x


class NotSoSimpleNet(nn.Module):
    def __init__(self):
        super(NotSoSimpleNet, self).__init__()

        self.cl1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1)
        self.at1 = nn.LeakyReLU()

        self.cl2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.at2 = nn.LeakyReLU()

        self.cl3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.at3 = nn.LeakyReLU()

        self.flatten = nn.Flatten(0, 2)

        self.fcn1 = nn.Linear(in_features=1352, out_features=1024)
        self.at4 = nn.LeakyReLU()

        self.fcn2 = nn.Linear(in_features=1024, out_features=512)
        self.at5 = nn.LeakyReLU()

        self.fcn3 = nn.Linear(in_features=512, out_features=256)
        self.at6 = nn.LeakyReLU()

        self.fcn4 = nn.Linear(in_features=256, out_features=128)
        self.at7 = nn.LeakyReLU()

        self.fcn5 = nn.Linear(in_features=128, out_features=128)
        self.at8 = nn.LeakyReLU()

        self.fcn6 = nn.Linear(in_features=128, out_features=64)
        self.at9 = nn.LeakyReLU()

        self.fcn7 = nn.Linear(in_features=64, out_features=32)
        self.at10 = nn.LeakyReLU()

        self.fcn8 = nn.Linear(in_features=32, out_features=16)
        self.at11 = nn.LeakyReLU()

        self.fcn9 = nn.Linear(in_features=16, out_features=4)
        self.at12 = nn.LeakyReLU()

        self.fcn10 = nn.Linear(in_features=4, out_features=3)


    def forward(self, x):
        x = self.cl1(x)
        x = self.at1(x)
        x = self.cl2(x)
        x = self.at2(x)
        x = self.cl3(x)
        x = self.at3(x)
        x = self.flatten(x)
        x = self.fcn1(x)
        x = self.at4(x)
        x = self.fcn2(x)
        x = self.at5(x)
        x = self.fcn3(x)
        x = self.at6(x)
        x = self.fcn4(x)
        x = self.at7(x)
        x = self.fcn5(x)
        x = self.at8(x)
        x = self.fcn6(x)
        x = self.at9(x)
        x = self.fcn7(x)
        x = self.at10(x)
        x = self.fcn8(x)
        x = self.at11(x)
        x = self.fcn9(x)
        x = self.at12(x)
        x = self.fcn10(x)

        return x


class NotSimpleNet(nn.Module):
    def __init__(self):
        super(NotSimpleNet, self).__init__()

        self.cl1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.at1 = nn.LeakyReLU()

        self.cl2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.at2 = nn.LeakyReLU()
        # shape: (25, 25, 4); Params: 296
        self.cl3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.at3 = nn.LeakyReLU()

        # shape: (13, 13, 8)
        self.flatten = nn.Flatten(0, 2)

        # shape: (1352); Params: 1385800
        self.fcn1 = nn.Linear(in_features=10816, out_features=1024)
        self.at4 = nn.LeakyReLU()

        # shape: (1024); Params: 263168
        self.fcn2 = nn.Linear(in_features=1024, out_features=512)
        self.at5 = nn.LeakyReLU()

        # shape: (256); Params: 16640
        self.fcn3 = nn.Linear(in_features=512, out_features=256)
        self.at6 = nn.LeakyReLU()

        # shape: (64); Params: 1088
        self.fcn4 = nn.Linear(in_features=256, out_features=128)
        self.at7 = nn.LeakyReLU()

        # shape: (16); Params: 80
        self.fcn5 = nn.Linear(in_features=128, out_features=128)
        self.at8 = nn.LeakyReLU()

        self.fcn6 = nn.Linear(in_features=128, out_features=64)
        self.at9 = nn.LeakyReLU()

        self.fcn7 = nn.Linear(in_features=64, out_features=32)
        self.at10 = nn.LeakyReLU()

        self.fcn8 = nn.Linear(in_features=32, out_features=16)
        self.at11 = nn.LeakyReLU()

        self.fcn9 = nn.Linear(in_features=16, out_features=4)
        self.at12 = nn.LeakyReLU()

        # Params: 16
        self.fcn10 = nn.Linear(in_features=4, out_features=3)

        # total (trainable) params: 1,667,184

    def forward(self, x):
        x = self.cl1(x)
        x = self.at1(x)
        x = self.cl2(x)
        x = self.at2(x)
        x = self.cl3(x)
        x = self.at3(x)
        x = self.flatten(x)
        x = self.fcn1(x)
        x = self.at4(x)
        x = self.fcn2(x)
        x = self.at5(x)
        x = self.fcn3(x)
        x = self.at6(x)
        x = self.fcn4(x)
        x = self.at7(x)
        x = self.fcn5(x)
        x = self.at8(x)
        x = self.fcn6(x)
        x = self.at9(x)
        x = self.fcn7(x)
        x = self.at10(x)
        x = self.fcn8(x)
        x = self.at11(x)
        x = self.fcn9(x)
        x = self.at12(x)
        x = self.fcn10(x)

        return x

