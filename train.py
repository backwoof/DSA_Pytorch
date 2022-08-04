import math

import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import argparse
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader

from model_new import RDN, Discriminator
from dsa_dataset import DSADataset

parser = argparse.ArgumentParser()

parser.add_argument('--LAMBDA', default=0.01, help='生成器损失平衡指数')
parser.add_argument('--img_dir', default='D:/JupyterNotebook/Unet_GAN_demo/data/membrane/train/image', help='img_dir')
parser.add_argument('--label_dir', default='D:/JupyterNotebook/Unet_GAN_demo/data/membrane/train/label', help='label_dir')

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=1, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=96, help='patch size')

parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')

parser.add_argument('--scale', type=int, default=1, help='scale output size /input size')

args = parser.parse_args()


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)


def get_dataset():
    d = DSADataset(args.img_dir, args.label_dir)
    dataloader = DataLoader(d, 1, True, num_workers=0)
    return dataloader


torch.backends.cudnn.enabled = False
def train(args):
    # 生成器
    my_model = RDN(args)
    # my_model.apply(weights_init_kaiming)
    my_model.cuda()
    # 判别器
    dis_model = Discriminator()
    # dis_model.apply(weights_init_kaiming)
    dis_model.cuda()
    dataloader = get_dataset()

    # 优化器
    G_opt = torch.optim.Adam(my_model.parameters(), lr=1e-5)
    D_opt = torch.optim.Adam(dis_model.parameters(), lr=1e-5)

    # 损失
    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        D_losses = []
        G_losses = []
        for batch, (img, target) in enumerate(dataloader):
            mini_batch = img.size()[0]
            img = Variable(img.cuda(), volatile=False)
            target = Variable(target.cuda())

            # 生成器产生分割结果
            gen_output = my_model(img)
            # 判别器对真实label的输出
            disc_real_output = dis_model(img, target)
            # 判别器对生成label的输出
            disc_generated_output = dis_model(img, gen_output)

            # 判别器loss
            real_loss = criterion(disc_real_output, Variable(torch.ones(disc_real_output.size()).cuda()))
            generated_loss = criterion(disc_generated_output, Variable(torch.zeros(disc_generated_output.size()).cuda()))
            D_loss = real_loss + generated_loss

            dis_model.zero_grad()
            D_loss.backward()
            D_opt.step()

            # 生成器loss
            gen_output = my_model(img)
            disc_generated_output = dis_model(img, gen_output)
            G_loss = criterion(disc_generated_output, torch.ones(disc_generated_output.size()).cuda()) + args.LAMBDA * criterion(target, gen_output)

            # 反向传播
            # dis_model.zero_grad()
            my_model.zero_grad()
            G_loss.backward()
            G_opt.step()

            # 损失值添加
            # D_losses.append(D_loss.data[0])
            # G_losses.append(G_loss.data[0])

        # if (epoch + 1) % 10 == 0:
            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f' % (epoch+1, args.epochs, batch+1, len(dataloader), D_loss.data[0], G_loss.data[0]))


if __name__ == '__main__':
    train(args)
