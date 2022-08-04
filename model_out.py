import torch
import torch.nn as nn
import torchvision.transforms as transforms

###
from dsa_dataset import DSADataset


class DCRBlock(nn.Module):
    def __init__(self, in_channel, kernelsize):
        super(DCRBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, int(kernelsize * 1.5), 3, 1, 1),
            nn.BatchNorm2d(int(kernelsize * 1.5))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(int(kernelsize * 2.5), kernelsize * 2, 3, 1, 1),
            nn.BatchNorm2d(kernelsize * 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(int(kernelsize * 5.5), kernelsize, 3, 1, 1),
            nn.BatchNorm2d(kernelsize)
        )

        self.conv = nn.Conv2d(kernelsize * 2, kernelsize, 3, 1, 1)

    def forward(self, x):
        concat1 = torch.cat([x, self.layer1(x)], dim=1)
        # 报错Given groups=1, weight of size [128, 128, 3, 3], expected input[1, 192, 512, 512] to have 128 channels,
        # but got 192 channels instead
        concat2 = torch.cat([x, concat1, self.layer2(concat1)], dim=1)
        # 原因：self.layer2(concat1)]中的concat1通道数不匹配
        res = torch.cat([x, self.layer3(concat2)], dim=1)
        res = self.conv(res)
        return res


class UnetDCR(nn.Module):
    def __init__(self, input_channel, kernelsize):  # input_channel=1,kernelsize=64
        super(UnetDCR, self).__init__()
        # 1 --> 64
        self.conv1 = nn.Conv2d(input_channel, kernelsize, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dcr1 = DCRBlock(kernelsize, kernelsize)
        self.maxPool1 = nn.MaxPool2d(2)
        # 64 --> 128
        self.conv2 = nn.Conv2d(kernelsize, kernelsize * 2, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.dcr2 = DCRBlock(kernelsize * 2, kernelsize * 2)
        self.maxPool2 = nn.MaxPool2d(2)
        # 128 --> 256
        self.conv3 = nn.Conv2d(kernelsize * 2, kernelsize * 4, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.dcr3 = DCRBlock(kernelsize * 4, kernelsize * 4)
        self.maxPool3 = nn.MaxPool2d(2)
        #
        self.up1 = nn.UpsamplingNearest2d(2)
        self.conv4 = nn.Conv2d(kernelsize * 4, kernelsize, 3, 1, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(kernelsize * 4, kernelsize, 3, 1, 1)
        self.relu5 = nn.ReLU(inplace=True)

        self.up2 = nn.UpsamplingNearest2d(2)
        self.conv6 = nn.Conv2d(kernelsize, kernelsize, 3, 1, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(kernelsize, kernelsize, 3, 1, 1)
        self.relu7 = nn.ReLU(inplace=True)

        self.up3 = nn.UpsamplingNearest2d(2)
        self.conv8 = nn.Conv2d(kernelsize, kernelsize, 3, 1, 1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(kernelsize, kernelsize, 3, 1, 1)
        self.relu9 = nn.ReLU(inplace=True)
        self.dcr4 = DCRBlock(kernelsize, kernelsize * 2)

        self.conv10 = nn.Conv2d(kernelsize, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1-->64
        conv_1 = self.relu1(self.conv1(x))
        dcr_1 = self.dcr1(conv_1)
        m_1 = self.maxPool1(dcr_1)
        # 64-->128
        conv_2 = self.relu2(self.conv2(m_1))
        dcr_2 = self.dcr2(conv_2)
        m_2 = self.maxPool1(dcr_2)
        # 128-->256
        conv_3 = self.relu3(self.conv3(m_2))
        dcr_3 = self.dcr3(conv_3)
        m_3 = self.maxPool1(dcr_3)
        # 256-->64
        u_3 = self.up1(m_3)
        conv_5 = self.relu4(self.conv4(u_3))
        conv_6 = self.relu5(self.conv5(dcr_3))
        con_3 = torch.cat([conv_5, conv_6], dim=1)

        u_2 = self.up2(con_3)
        conv_3 = self.relu6(self.conv6(u_2))
        conv_4 = self.relu7(self.conv7(dcr_2))
        con_2 = torch.cat([conv_3, conv_4], dim=1)

        u_1 = self.up3(con_2)
        conv_1 = self.relu8(self.conv8(u_1))
        conv_2 = self.relu9(self.conv9(dcr_1))
        con = torch.cat([conv_1, conv_2], dim=1)

        ddcr_1 = self.dcr4(con)

        convfinal = self.sigmoid(self.conv10(ddcr_1))
        return convfinal


def main():
    img_dir = 'D:/JupyterNotebook/Unet_GAN_demo/data/membrane/train/image'
    label_dir = 'D:/JupyterNotebook/Unet_GAN_demo/data/membrane/train/label'
    d = DSADataset(img_dir, label_dir)
    dataitem = iter(d)
    imgs, labels = next(dataitem)
    # [512, 512] --> [1, 1, 512, 512]
    imgs = torch.unsqueeze(imgs, 0)  # 在下标0处增加维度
    imgs = torch.unsqueeze(imgs, 1)  # 在下标1处增加维度
    print(imgs.size())
    model = UnetDCR(1, 64)
    return model(imgs)


if __name__ == '__main__':
    main()
