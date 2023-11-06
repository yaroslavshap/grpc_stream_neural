import torch
import torchvision as tv
import matplotlib as plt
import time
from torch import nn
import torch.nn.functional as F


class Subnet1(nn.Module):
    def __init__(self):
        super(Subnet1, self).__init__()

        self.hidden_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.LeakyReLU()
        )
        self.hidden_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.LeakyReLU()
        )
        self.hidden_3 = nn.Sequential(
            nn.Conv3d(96, 128, kernel_size=(2, 7, 7), stride=(2, 2, 2), padding=(0, 3, 3)),
            nn.LeakyReLU()
        )

        self.hidden_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.LeakyReLU()
        )

        self.conv_5 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.activation_conv_5 = nn.LeakyReLU()
        self.maxpool_conv_5 = nn.MaxPool2d((2, 2))

        self.hidden_6 = nn.AdaptiveAvgPool2d((1, 1))

        self.hidden_7 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 64 * 128)  # 64 * 128
        )

    def forward_once(self, x):
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        return x

    def forward(self, input1, input2):
        # siames part
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        merged_out = torch.stack([out1, out2], dim=2)
        # paralax learning part
        layer3 = self.hidden_3(merged_out)[:, :, 0]
        layer4 = self.hidden_4(layer3)
        # hidden 5
        conv5 = self.conv_5(layer4)
        cat5 = torch.cat([layer3, conv5], dim=1)  # [10, 256, 32, 64]
        activ5 = self.activation_conv_5(cat5)  # [10, 256, 32, 64]
        pool5 = self.maxpool_conv_5(activ5)  # [10, 256, 1, 1]
        # hidden 6
        hidden6 = self.hidden_6(pool5)
        hidden7 = self.hidden_7(hidden6)
        x = hidden7.view(-1, 1, 64, 128)  # [10, 1, 64, 128]
        return x


class PyramydPool(nn.Module):
    def __init__(self):
        super(PyramydPool, self).__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=1, padding=0),
                                     nn.Conv2d(1, 32, (3, 3), 1, 1),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32))
        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(1, 1)),
                                     nn.Conv2d(1, 32, (3, 3), 1, 1),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32))
        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(1, 1)),
                                     nn.Conv2d(1, 32, (3, 3), 1, 1),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32))
        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(1, 1)),
                                     nn.Conv2d(1, 32, (3, 3), 1, 1),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32))
        self.lastconv = nn.Sequential(nn.Conv2d(128, 4, (3, 3), (1, 1), 1),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        out_branch1 = self.branch1(x)
        out_branch1 = F.interpolate(out_branch1, (x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        out_branch2 = self.branch2(x)
        out_branch2 = F.interpolate(out_branch2, (x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        out_branch3 = self.branch3(x)
        out_branch3 = F.interpolate(out_branch3, (x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        out_branch4 = self.branch4(x)
        out_branch4 = F.interpolate(out_branch4, (x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        concatination_block = torch.cat((out_branch1, out_branch2, out_branch3, out_branch4), dim=1)
        out_block = self.lastconv(concatination_block)
        out_block = F.interpolate(out_block, (128, 256), mode='bilinear', align_corners=True)
        return out_block


class SubNet2(nn.Module):
    def __init__(self):
        super(SubNet2, self).__init__()

        self.netpool = PyramydPool()

        self.hidden_1 = nn.Sequential(
            nn.Conv2d(3, 60, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(60),
            nn.LeakyReLU()
        )

        self.hidden_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(9, 9), stride=(1, 1), padding=4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.hidden_3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(7, 7), stride=(1, 1), padding=3),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )

        self.conv4 = nn.Conv2d(96, 64, (5, 5), (1, 1), 2)
        self.batch4 = nn.BatchNorm2d(64)
        self.activ4 = nn.LeakyReLU()

        self.hidden_5 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU()
        )

        self.conv_for_cat_6 = nn.Conv2d(96, 64, (3, 3), (1, 1), 1)
        self.batch_for_cat_6 = nn.BatchNorm2d(64)
        self.activation_conv_6 = nn.LeakyReLU()

        self.hidden_7 = nn.Sequential(
            nn.ConvTranspose2d(192, 1, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, right, mask):
        hidden_1 = self.hidden_1(right)
        mask_up = self.netpool(mask)

        cat_mask_right = torch.cat([hidden_1, mask_up], dim=1)

        hidden_2 = self.hidden_2(cat_mask_right)
        hidden_3 = self.hidden_3(hidden_2)
        # hidden 4
        conv4 = self.conv4(hidden_3)
        batch4 = self.batch4(conv4)
        out = torch.cat([hidden_2, batch4], dim=1)
        activ4 = self.activ4(out)
        # hidden 5
        hidden_5 = self.hidden_5(activ4)
        # hidden 6
        canv6 = self.conv_for_cat_6(hidden_5)
        batch6 = self.batch_for_cat_6(canv6)
        out = torch.cat([activ4, batch6], dim=1)
        activ_6 = self.activation_conv_6(out)
        hidden_7 = self.hidden_7(activ_6)

        return hidden_7


class CommonNet(nn.Module):
    def __init__(self):
        super(CommonNet, self).__init__()
        self.net1 = Subnet1()
        self.net2 = SubNet2()

    def forward(self, left, right):
        mask = self.net1(left, right)
        x = self.net2(right, mask)

        return x
