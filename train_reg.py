import os
import argparse
import visdom
import time
import numpy as np

import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from model import Generator, Discriminator
from data import ConvertCapVec, ReadFromVec
from storage_utils import save_statistics


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--caption_root', type=str, required=True,
                    help='root directory that contains captions')
parser.add_argument('--trainclasses_file', type=str, required=True,
                    help='text file that contains training classes')
parser.add_argument('--save_filename_G', type=str, required=True,
                    help='checkpoint file of generator')
parser.add_argument('--save_filename_D', type=str, required=True,
                    help='checkpoint file of discriminator')
parser.add_argument('--save_filename_stats', type=str, required=True,
                    help='stats file path')
parser.add_argument('--log_interval', type=int, default=10,
                    help='the number of iterations (default: 10)')
parser.add_argument('--num_threads', type=int, default=8,
                    help='number of threads for fetching data (default: 8)')
parser.add_argument('--num_epochs', type=int, default=600,
                    help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate (dafault: 0.0002)')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='learning rate decay (dafault: 0.5)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='beta1 for Adam optimizer (dafault: 0.5)')
parser.add_argument('--lambda_cond_loss', type=float, default=10,
                    help='lambda of conditional loss (default: 10)')
parser.add_argument('--lambda_recon_loss', type=float, default=0.2,
                    help='lambda of reconstruction loss (default: 0.2)')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--visdom_server', type=str,
                    help='Use visdom server')
parser.add_argument('--gamma_0', type=float,
                    help='Initial regularisation constant')

args = parser.parse_args()

arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
print(arg_str)


def label_like(label, x):
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v


def zeros_like(x):
    return label_like(0, x)


def ones_like(x):
    return label_like(1, x)


def gradient_penalties(d1_logits, d1_arg, d2_logits, d2_arg):
    batch_size = d1_logits.shape[0]
    d1 = torch.sigmoid(d1_logits)
    d2 = torch.sigmoid(d1_logits)

    grad_d1_logits = grad(d1_logits, d1_arg, grad_outputs=ones_like(d1_logits),
                          create_graph=True, retain_graph=True)[0]
    grad_d2_logits = grad(d2_logits, d2_arg, grad_outputs=ones_like(d2_logits),
                          create_graph=True, retain_graph=True)[0]
    grad_d1_norm = torch.norm(grad_d1_logits.view(batch_size, -1), dim=1)
    grad_d2_norm = torch.norm(grad_d2_logits.view(batch_size, -1), dim=1)

    reg_d1 = torch.mul((1.0 - d1) ** 2, grad_d1_norm ** 2)
    reg_d2 = torch.mul(d2 ** 2, grad_d2_norm ** 2)
    reg = torch.mean(reg_d1 + reg_d2)
    return reg


if __name__ == '__main__':
    if not args.no_cuda and not torch.cuda.is_available():
        print('Warning: cuda is not available on this machine.')
        args.no_cuda = True
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    caption_root = args.caption_root.split('/')[-1]
    if (caption_root + '_vec') not in os.listdir(args.caption_root.replace(caption_root, '')):
        raise RuntimeError('Caption data was not prepared. Please run preprocess_caption.py.')

    if not os.path.exists(os.path.dirname(args.save_filename_G)):
        os.makedirs(os.path.dirname(args.save_filename_G))
    if not os.path.exists(os.path.dirname(args.save_filename_D)):
        os.makedirs(os.path.dirname(args.save_filename_D))

    print('Loading a dataset...')
    train_data = ReadFromVec(args.img_root,
                             args.caption_root,
                             args.trainclasses_file,
                             transforms.Compose([
                                transforms.Resize(136),
                                transforms.RandomCrop(128),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor()]))

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_threads)

    G = Generator()
    D = Discriminator()
    G, D = G.to(device), D.to(device)

    g_optimizer = torch.optim.Adam(G.parameters(),
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(),
                                   lr=args.learning_rate, betas=(args.momentum, 0.999))
    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, args.lr_decay)
    d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, 100, args.lr_decay)

    if args.visdom_server:
        vis = visdom.Visdom(server=args.visdom_server)

    total_losses = {'Epoch': [], 'D_real': [], 'D_real_c': [], 'D_fake': [], 'G_fake': [],
                    'G_fake_c': [], 'G_recon': [], 'KLD': [], 'D_Reg': [], 'Time': []}

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()

        avg_D_real_loss = 0
        avg_D_real_c_loss = 0
        avg_D_fake_loss = 0
        avg_G_fake_loss = 0
        avg_G_fake_c_loss = 0
        avg_G_recon_loss = 0
        avg_kld = 0
        avg_d_reg = 0

        gamma = args.gamma_0 * 0.01 ** (epoch / (args.num_epochs - 1))
        for i, (img, txt, len_txt) in enumerate(train_loader):
            img, txt, len_txt = img.to(device), txt.to(device), len_txt.to(device)
            img = img.mul(2).sub(1)
            img.requires_grad = True

            # BTC to TBC
            txt = txt.transpose(1, 0)
            # negative text
            txt_m = torch.cat((txt[:, -1, :].unsqueeze(1), txt[:, :-1, :]), 1)
            len_txt_m = torch.cat((len_txt[-1].unsqueeze(0), len_txt[:-1]))

            # UPDATE DISCRIMINATOR
            D.zero_grad()

            # real images
            real_logit, real_c_prob, real_c_prob_n = D(img, txt, len_txt, negative=True)

            real_loss = F.binary_cross_entropy_with_logits(real_logit, ones_like(real_logit))
            avg_D_real_loss += real_loss.item()

            real_c_loss = (F.binary_cross_entropy(real_c_prob, ones_like(real_c_prob)) + \
                F.binary_cross_entropy(real_c_prob_n, zeros_like(real_c_prob_n))) / 2
            avg_D_real_c_loss += real_c_loss.item()

            real_loss = real_loss + args.lambda_cond_loss * real_c_loss

            # synthesized images
            fake, _ = G(img, (txt_m, len_txt_m))
            fake_logit, _ = D(fake, txt_m, len_txt_m)

            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, zeros_like(fake_logit))
            avg_D_fake_loss += fake_loss.item()

            # regularisation
            d_reg = (gamma / 2.0) * gradient_penalties(real_logit, img, fake_logit, fake)
            avg_d_reg += d_reg.item()

            loss = real_loss + fake_loss + d_reg
            loss.backward()

            d_optimizer.step()

            # UPDATE GENERATOR
            G.zero_grad()

            fake, (z_mean, z_log_stddev) = G(img, (txt_m, len_txt_m))

            kld = torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += 0.5 * kld.item()

            fake_logit, fake_c_prob = D(fake, txt_m, len_txt_m)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ones_like(fake_logit))
            avg_G_fake_loss += fake_loss.item()
            fake_c_loss = F.binary_cross_entropy(fake_c_prob, ones_like(fake_c_prob))
            avg_G_fake_c_loss += fake_c_loss.item()

            G_loss = fake_loss + args.lambda_cond_loss * fake_c_loss + 0.5 * kld

            G_loss.backward()

            # reconstruction for matching input
            recon, (z_mean, z_log_stddev) = G(img, (txt, len_txt))

            kld = torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += 0.5 * kld.item()

            recon_loss = F.l1_loss(recon, img)
            avg_G_recon_loss += recon_loss.item()

            G_loss = args.lambda_recon_loss * recon_loss + 0.5 * kld

            G_loss.backward()

            g_optimizer.step()

            if i % args.log_interval == 0:
                print('Epoch [%03d/%03d], Iter [%03d/%03d], D_real: %.4f, D_real_c: %.4f, D_fake: %.4f, G_fake: %.4f, G_fake_c: %.4f, G_recon: %.4f, D_Reg: %.4f, KLD: %.4f'
                    % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
                        avg_D_real_c_loss / (i + 1), avg_D_fake_loss / (i + 1),
                        avg_G_fake_loss / (i + 1), avg_G_fake_c_loss / (i + 1),
                        avg_G_recon_loss / (i + 1), avg_d_reg / (i + 1), avg_kld / (i + 1)))

        d_lr_scheduler.step()
        g_lr_scheduler.step()

        epoch_elapsed_time = time.time() - epoch_start_time
        epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)

        print('Epoch [%03d/%03d], Iter [%03d/%03d], D_real: %.4f, D_real_c: %.4f, D_fake: %.4f, G_fake: %.4f, G_fake_c: %.4f, G_recon: %.4f, D_Reg: %.4f, KLD: %.4f, Time: %s'
            % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
                avg_D_real_c_loss / (i + 1), avg_D_fake_loss / (i + 1),
                avg_G_fake_loss / (i + 1), avg_G_fake_c_loss / (i + 1),
                avg_G_recon_loss / (i + 1), avg_d_reg / (i + 1), avg_kld / (i + 1), epoch_elapsed_time))

        total_losses['Epoch'].append(epoch + 1)
        total_losses['D_real'].append(avg_D_real_loss / (i + 1))
        total_losses['D_real_c'].append(avg_D_real_c_loss / (i + 1))
        total_losses['D_fake'].append(avg_D_fake_loss / (i + 1))
        total_losses['G_fake'].append(avg_G_fake_loss / (i + 1))
        total_losses['G_fake_c'].append(avg_G_fake_c_loss / (i + 1))
        total_losses['G_recon'].append(avg_G_recon_loss / (i + 1))
        total_losses['KLD'].append(avg_kld / (i + 1))
        total_losses['D_Reg'].append(avg_d_reg / (i + 1))
        total_losses['Time'].append(epoch_elapsed_time)

        save_statistics(args.save_filename_stats, 'summary.csv', total_losses, epoch,
                        continue_from_mode=(i > 0))

        if args.visdom_server:
            img_vis = img.mul(0.5).add(0.5)
            vis.images(img_vis.cpu().detach().numpy(), nrow=4, opts=dict(title='original'))
            fake_vis = fake.mul(0.5).add(0.5)
            vis.images(fake_vis.cpu().detach().numpy(), nrow=4, opts=dict(title='generated'))

        torch.save(G.state_dict(), args.save_filename_G)
        torch.save(D.state_dict(), args.save_filename_D)
