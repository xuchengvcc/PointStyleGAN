# encoding=utf-8
import json
import numpy as np
import math
import sys
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import logging
import random
# import imageio
# add for shape-preserving Loss
from Common_snow.point_operation import normalize_point_cloud
from Common_snow.loss_utils import get_local_pair, AverageValueMeter, dist_simple
from Common_snow import loss_utils
from tensorboardX import SummaryWriter
from Common_snow.visu_utils import plot_pcd_three_views, point_cloud_three_views, plot_pcd_multi_rows
from tqdm import tqdm
from Generation_snow_style_S_adapt_stylegan.Generator_2_const import Generator
from Generation_snow_style_S_adapt_stylegan.Discriminator_stylegan_2_muti import Discriminator
from metrics.evaluation_metrics import compute_all_metrics
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from Common_snow.network_utils import *
import copy
from pprint import pprint
from visualization import plot

cudnn.benchnark = True

from CRN_dataset import CRNShapeNet
import os.path as osp

from datetime import datetime
from tqdm import tqdm
# from time import time
from Generation_snow_style_S_adapt_stylegan.H5DataLoader_4point import H5DataLoader
from models.utils import fps_subsample

from data_utils.hdf5_loader import get_data_loaders
from visualization.plot import plot_multi_scale_pcd,normalize_point_cloud,plot_multi_scale_up_pcd,plot_multi_scale_up_diff_pcd

# seed = 123
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class Model(object):
    def __init__(self, opts):
        self.opts = opts

    def backup(self):
        if self.opts.phase == 'train':
            source_folder = os.path.join(os.getcwd(), "Generation_snow_style_S_adapt_stylegan")
            common_folder = os.path.join(os.getcwd(), "Common_snow")
            os.system("cp %s/Generator_2_const.py '%s/Generator_2_const.py'" % (source_folder, self.opts.log_dir))
            os.system("cp %s/Discriminator_stylegan_2_muti.py '%s/Discriminator_stylegan_2_muti.py'" % (source_folder, self.opts.log_dir))
            os.system("cp %s/model_PS_2_const_dis.py '%s/model_PS_2_const_dis.py'" % (source_folder, self.opts.log_dir))
            os.system("cp %s/loss_utils.py '%s/loss_utils.py'" % (common_folder, self.opts.log_dir))
            os.system("cp %s/H5DataLoader_4point.py '%s/H5DataLoader_4point.py'" % (source_folder, self.opts.log_dir))

    def build_model(self):
        """ Models """

        self.G = Generator(self.opts, up_factors=[2])
        self.G_ema = Generator(self.opts, up_factors=[2])
        self.D1 = Discriminator(self.opts)
        self.D2 = Discriminator(self.opts)
        self.D3 = Discriminator(self.opts)
        self.D4 = Discriminator(self.opts)
        self.D5 = Discriminator(self.opts)
        self.D6 = Discriminator(self.opts)
        self.D7 = Discriminator(self.opts)
        self.D8 = Discriminator(self.opts)
        self.D9 = Discriminator(self.opts)

        self.multi_gpu = False

        self.G.cuda()
        self.G_ema.cuda()
        self.accumulate(self.G_ema, self.G, 0.0)
        # self.D1.cuda()
        self.D2.cuda()
        self.D3.cuda()
        self.D4.cuda()
        self.D5.cuda()
        self.D6.cuda()
        self.D7.cuda()
        self.D8.cuda()
        self.D9.cuda()

        """ Training """

        beta1 = 0.5
        beta2 = 0.99
        self.optimizerG = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.opts.lr_g,
                                     betas=(beta1, beta2))
        # self.optimizerD1 = optim.Adam(filter(lambda p: p.requires_grad, self.D1.parameters()), lr=self.opts.lr_d,
        #                               betas=(beta1, beta2))
        self.optimizerD2 = optim.Adam(filter(lambda p: p.requires_grad, self.D2.parameters()), lr=self.opts.lr_d,
                                      betas=(beta1, beta2))
        self.optimizerD3 = optim.Adam(filter(lambda p: p.requires_grad, self.D3.parameters()), lr=self.opts.lr_d,
                                      betas=(beta1, beta2))
        self.optimizerD4 = optim.Adam(filter(lambda p: p.requires_grad, self.D4.parameters()), lr=self.opts.lr_d,
                                      betas=(beta1, beta2))
        self.optimizerD5 = optim.Adam(filter(lambda p: p.requires_grad, self.D5.parameters()), lr=self.opts.lr_d,
                                      betas=(beta1, beta2))
        self.optimizerD6 = optim.Adam(filter(lambda p: p.requires_grad, self.D6.parameters()), lr=self.opts.lr_d,
                                      betas=(beta1, beta2))
        self.optimizerD7 = optim.Adam(filter(lambda p: p.requires_grad, self.D7.parameters()), lr=self.opts.lr_d,
                                      betas=(beta1, beta2))
        self.optimizerD8 = optim.Adam(filter(lambda p: p.requires_grad, self.D8.parameters()), lr=self.opts.lr_d,
                                      betas=(beta1, beta2))
        self.optimizerD9 = optim.Adam(filter(lambda p: p.requires_grad, self.D9.parameters()), lr=self.opts.lr_d,
                                      betas=(beta1, beta2))

        if self.opts.lr_decay:
            if self.opts.use_sgd:
                self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, self.opts.max_epoch,
                                                                              eta_min=self.opts.lr_g)
            else:
                self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizerG, step_size=self.opts.lr_decay_feq,
                                                                   gamma=self.opts.lr_decay_rate)
        else:
            self.scheduler_G = None

        if self.opts.lr_decay:
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizerD, step_size=self.opts.lr_decay_feq,
                                                               gamma=self.opts.lr_decay_rate)
        else:
            self.scheduler_D = None

        if self.opts.restore:
            could_load, save_epoch = self.load(self.opts.save_dir)
            if could_load:
                self.opts.start_epoch = save_epoch
                print(" [*] Load SUCCESS")

            # self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
            self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')
            self.LOG_FOUT.write(str(self.opts) + '\n')
        else:
            print('training...')
            self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')
            self.LOG_FOUT.write(str(self.opts) + '\n')

        print('# generator parameters:', sum(param.numel() for param in self.G.parameters()))
        print('# discriminator1 parameters:', sum(param.numel() for param in self.D9.parameters()))

    def noise_generator(self, bs=1, masks=None):

        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(0, self.opts.nv, (bs, self.opts.np, self.opts.nz))
            else:
                noise = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))

            if self.opts.n_mix and random.random() < 0.5:
                noise2 = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))
                for i in range(bs):
                    id = np.random.randint(0, self.opts.np)
                    idx = np.argsort(self.ball_dist[id])[::1]
                    # idx = np.arange(self.opts.np)
                    # np.random.shuffle(idx)
                    num = int(max(random.random(), 0.1) * self.opts.np)
                    noise[i, idx[:num]] = noise2[i]
        else:
            noise = np.zeros((bs, self.opts.np, self.opts.nz))
            for i in range(masks.shape[0]):
                mask = masks[i]
                unique_mask = np.unique(mask)
                for j in unique_mask:
                    noise_once = np.random.normal(0, 0.2, (1, self.opts.nz))
                    idx = np.where(mask == j)
                    noise[i, idx] = idx

        sim_noise = Variable(torch.Tensor(noise)).cuda()

        return sim_noise
    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    def train(self):
        
        global epoch
        self.build_model()
        self.get_D_weight()
        self.backup()
        self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
        self.log_string('PARAMETER ...')
        with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments
        pprint(self.opts)

        if not os.path.exists(self.opts.tensorboard_path):
            os.makedirs(self.opts.tensorboard_path)

        self.writer = SummaryWriter(self.opts.tensorboard_path)
        '''DATA LOADING'''
        self.log_string('Load dataset ...')
        # self.train_dataset = H5DataLoader(self.opts, augment=self.opts.augment)  # (6778,2048,3)
        # self.dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.opts.bs, shuffle=True,
        #                                               num_workers=int(self.opts.workers), drop_last=True,
        #                                               pin_memory=True)
        dataloader, x_train_var = get_data_loaders(self.opts)
        self.dataloader = dataloader['train_loader']
        # loop for epoch
        start_time = time.time()
        d_avg_meter = AverageValueMeter()
        g_avg_meter = AverageValueMeter()
        real_acc_avg_meter = AverageValueMeter()
        fake_acc_avg_meter = AverageValueMeter()

        sfm = nn.Softmax(dim=1)
        kl_loss = nn.KLDivLoss()
        sim = nn.CosineSimilarity()

        global_step = 0

        d_para = 1.0
        g_para = 1.0


        for epoch in range(self.opts.start_epoch, self.opts.max_epoch + 1):
            # self.D1.train()
            self.D2.train()
            self.D3.train()
            self.D4.train()
            self.D5.train()
            self.D6.train()
            self.D7.train()
            self.D8.train()
            self.D9.train()
            self.G.train()

            step_d = 0
            step_g = 0
            for idx, data in tqdm(enumerate(self.dataloader, 0), total=len(self.dataloader)):
                self.set_grad(1)
                gt = data['train_points'].cuda()
                # gt = data.cuda()
                fps_gt = fps_subsample(gt, 2048)
                points2 = fps_gt[:,:16,:]
                points3 = fps_gt[:,:32,:]
                points4 = fps_gt[:,:64,:]
                points5 = fps_gt[:,:128,:]
                points6 = fps_gt[:,:256,:]
                points7 = fps_gt[:,:512,:]
                points8 = fps_gt[:,:1024,:]
                points9 = fps_gt

                real_point2 = Variable(points2, requires_grad=True)
                real_point3 = Variable(points3, requires_grad=True)
                real_point4 = Variable(points4, requires_grad=True)
                real_point5 = Variable(points5, requires_grad=True)
                real_point6 = Variable(points6, requires_grad=True)
                real_point7 = Variable(points7, requires_grad=True)
                real_point8 = Variable(points8, requires_grad=True)
                real_point9 = Variable(points9, requires_grad=True)

                real_points2 = real_point2.transpose(2, 1).cuda()
                real_points3 = real_point3.transpose(2, 1).cuda()
                real_points4 = real_point4.transpose(2, 1).cuda()
                real_points5 = real_point5.transpose(2, 1).cuda()
                real_points6 = real_point6.transpose(2, 1).cuda()
                real_points7 = real_point7.transpose(2, 1).cuda()
                real_points8 = real_point8.transpose(2, 1).cuda()
                real_points9 = real_point9.transpose(2, 1).cuda()

                # -----------------------------------train D2-----------------------------------

                self.optimizerD2.zero_grad()

                z = self.noise_generator(bs=self.opts.bs)
                d_fake_preds, _ = self.G(z)  # [bs,3,2048]

                d_fake_preds2 = d_fake_preds[1].permute(0, 2, 1).contiguous()
                d_fake_preds2 = d_fake_preds2.detach()

                d_real_logit2 = self.D2(real_points2)
                d_fake_logit2 = self.D2(d_fake_preds2)

                lossD2, info = loss_utils.dis_loss(d_real_logit2, d_fake_logit2, gan=self.opts.gan,
                                                   noise_label=self.opts.flip_d)

                # if not torch.any(torch.isnan(lossD2)):
                lossD2.backward()
                self.optimizerD2.step()

                # -----------------------------------train D3-----------------------------------
                self.set_grad(2)

                self.optimizerD3.zero_grad()

                d_fake_preds3 = d_fake_preds[2].permute(0, 2, 1).contiguous()
                d_fake_preds3 = d_fake_preds3.detach()

                d_real_logit3 = self.D3(real_points3)
                d_fake_logit3 = self.D3(d_fake_preds3)

                lossD3, info = loss_utils.dis_loss(d_real_logit3, d_fake_logit3, gan=self.opts.gan,
                                                   noise_label=self.opts.flip_d)

                # if not torch.any(torch.isnan(lossD3)):
                lossD3.backward()
                self.optimizerD3.step()

                # -----------------------------------train D4-----------------------------------
                self.set_grad(3)

                self.optimizerD4.zero_grad()

                d_fake_preds4 = d_fake_preds[3].permute(0, 2, 1).contiguous()
                d_fake_preds4 = d_fake_preds4.detach()

                d_real_logit4 = self.D4(real_points4)
                d_fake_logit4 = self.D4(d_fake_preds4)

                lossD4, info = loss_utils.dis_loss(d_real_logit4, d_fake_logit4, gan=self.opts.gan,
                                                   noise_label=self.opts.flip_d)

                # if not torch.any(torch.isnan(lossD4)):
                lossD4.backward()
                self.optimizerD4.step()

                # -----------------------------------train D5-----------------------------------
                self.set_grad(4)
                self.optimizerD5.zero_grad()

                d_fake_preds5 = d_fake_preds[4].permute(0, 2, 1).contiguous()
                d_fake_preds5 = d_fake_preds5.detach()

                d_real_logit5 = self.D5(real_points5)
                d_fake_logit5 = self.D5(d_fake_preds5)

                lossD5, info = loss_utils.dis_loss(d_real_logit5, d_fake_logit5, gan=self.opts.gan,
                                                   noise_label=self.opts.flip_d)

                # if not torch.any(torch.isnan(lossD5)):
                lossD5.backward()
                self.optimizerD5.step()

                # -----------------------------------train D6-----------------------------------
                self.set_grad(5)
                self.optimizerD6.zero_grad()

                d_fake_preds6 = d_fake_preds[5].permute(0, 2, 1).contiguous()
                d_fake_preds6 = d_fake_preds6.detach()

                d_real_logit6 = self.D6(real_points6)
                d_fake_logit6 = self.D6(d_fake_preds6)

                lossD6, info = loss_utils.dis_loss(d_real_logit6, d_fake_logit6, gan=self.opts.gan,
                                                   noise_label=self.opts.flip_d)

                # if not torch.any(torch.isnan(lossD6)):
                lossD6.backward()
                self.optimizerD6.step()
                # -----------------------------------train D7-----------------------------------
                self.set_grad(6)
                self.optimizerD7.zero_grad()

                d_fake_preds7 = d_fake_preds[6].permute(0, 2, 1).contiguous()
                d_fake_preds7 = d_fake_preds7.detach()

                d_real_logit7 = self.D7(real_points7)
                d_fake_logit7 = self.D7(d_fake_preds7)

                lossD7, info = loss_utils.dis_loss(d_real_logit7, d_fake_logit7, gan=self.opts.gan,
                                                   noise_label=self.opts.flip_d)

                # if not torch.any(torch.isnan(lossD7)):
                lossD7.backward()
                self.optimizerD7.step()
                # -----------------------------------train D8-----------------------------------
                self.set_grad(7)
                self.optimizerD8.zero_grad()

                d_fake_preds8 = d_fake_preds[7].permute(0, 2, 1).contiguous()
                d_fake_preds8 = d_fake_preds8.detach()

                d_real_logit8 = self.D8(real_points8)
                d_fake_logit8 = self.D8(d_fake_preds8)

                lossD8, info = loss_utils.dis_loss(d_real_logit8, d_fake_logit8, gan=self.opts.gan,
                                                   noise_label=self.opts.flip_d)

                # if not torch.any(torch.isnan(lossD8)):
                lossD8.backward()
                self.optimizerD8.step()
                # -----------------------------------train D9-----------------------------------
                self.set_grad(8)
                self.optimizerD9.zero_grad()

                d_fake_preds9 = d_fake_preds[-1].permute(0, 2, 1).contiguous()
                d_fake_preds9 = d_fake_preds9.detach()

                d_real_logit9 = self.D9(real_points9)
                d_fake_logit9 = self.D9(d_fake_preds9)

                lossD9, info = loss_utils.dis_loss(d_real_logit9, d_fake_logit9, gan=self.opts.gan,
                                                   noise_label=self.opts.flip_d)

                # if not torch.any(torch.isnan(lossD9)):
                lossD9.backward()
                self.optimizerD9.step()

                # -----------------------------------train G-----------------------------------
                self.set_grad(0)
                self.optimizerG.zero_grad()

                z = self.noise_generator(bs=self.opts.bs)
                g_fake_pred, up_points = self.G(z)
                g_fake_preds2 = g_fake_pred[1].permute(0, 2, 1).contiguous()
                g_fake_preds3 = g_fake_pred[2].permute(0, 2, 1).contiguous()
                g_fake_preds4 = g_fake_pred[3].permute(0, 2, 1).contiguous()
                g_fake_preds5 = g_fake_pred[4].permute(0, 2, 1).contiguous()
                g_fake_preds6 = g_fake_pred[5].permute(0, 2, 1).contiguous()
                g_fake_preds7 = g_fake_pred[6].permute(0, 2, 1).contiguous()
                g_fake_preds8 = g_fake_pred[7].permute(0, 2, 1).contiguous()
                g_fake_preds9 = g_fake_pred[-1].permute(0, 2, 1).contiguous()

                g_real_logit2 = self.D2(real_points2)
                g_fake_logit2, feat_dis2 = self.D2(g_fake_preds2, return_feats=True)
                g_real_logit3 = self.D3(real_points3)
                g_fake_logit3, feat_dis3 = self.D3(g_fake_preds3, return_feats=True)
                g_real_logit4 = self.D4(real_points4)
                g_fake_logit4, feat_dis4 = self.D4(g_fake_preds4, return_feats=True)
                g_real_logit5 = self.D5(real_points5)
                g_fake_logit5, feat_dis5 = self.D5(g_fake_preds5, return_feats=True)
                g_real_logit6 = self.D6(real_points6)
                g_fake_logit6, feat_dis6 = self.D6(g_fake_preds6, return_feats=True)
                g_real_logit7 = self.D7(real_points7)
                g_fake_logit7, feat_dis7 = self.D7(g_fake_preds7, return_feats=True)
                g_real_logit8 = self.D8(real_points8)
                g_fake_logit8, feat_dis8 = self.D8(g_fake_preds8, return_feats=True)
                g_real_logit9 = self.D9(real_points9)
                g_fake_logit9, feat_dis9 = self.D9(g_fake_preds9, return_feats=True)

                # for pair1 in range(self.opts.feat_const_batch):
                #     tmp = 0
                #     for pair2 in range(self.opts.feat_const_batch):
                #         if pair1 != pair2:
                #             anchor_feat = torch.unsqueeze(z[pair1].reshape(-1), 0)
                #             compare_feat = torch.unsqueeze(z[pair2].reshape(-1), 0)
                #             dist_source[pair1, tmp] = sim(anchor_feat, compare_feat)
                #             tmp += 1
                # dist_source = sfm(dist_source)
                # dist_target = torch.zeros([self.opts.feat_const_batch, self.opts.feat_const_batch - 1]).cuda()
                #
                # for pair1 in range(self.opts.feat_const_batch):
                #     tmp = 0
                #     for pair2 in range(self.opts.feat_const_batch):
                #         if pair1 != pair2:
                #             anchor_feat = torch.unsqueeze(feat_dis9[pair1].reshape(-1), 0)
                #             compare_feat = torch.unsqueeze(feat_dis9[pair2].reshape(-1), 0)
                #             dist_target[pair1, tmp] = sim(anchor_feat, compare_feat)
                #             tmp += 1
                # dist_target = sfm(dist_target)
                # rel_loss = self.opts.kl_wt * kl_loss(torch.log(dist_target), dist_source)  # distance consistency loss
                #
                dist_source = torch.zeros([self.opts.feat_const_batch, self.opts.feat_const_batch - 1]).cuda()

                # _1
                for pair1 in range(self.opts.feat_const_batch):
                    tmp = 0
                    for pair2 in range(self.opts.feat_const_batch):
                        if pair1 != pair2:
                            anchor_feat = torch.unsqueeze(z[pair1].reshape(-1), 0)
                            compare_feat = torch.unsqueeze(z[pair2].reshape(-1), 0)
                            dist_source[pair1, tmp] = sim(anchor_feat, compare_feat)
                            tmp += 1
                dist_source = sfm(dist_source)

                dist_target = torch.zeros([self.opts.feat_const_batch, self.opts.feat_const_batch - 1]).cuda()

                # _2
                for pair1 in range(self.opts.feat_const_batch):
                    tmp = 0
                    for pair2 in range(self.opts.feat_const_batch):
                        if pair1 != pair2:
                            anchor_feat = torch.unsqueeze(feat_dis9[pair1].reshape(-1), 0)
                            compare_feat = torch.unsqueeze(feat_dis9[pair2].reshape(-1), 0)
                            dist_target[pair1, tmp] = sim(anchor_feat, compare_feat)
                            tmp += 1
                dist_target = sfm(dist_target)
                rel_loss = self.opts.kl_wt * kl_loss(torch.log(dist_target), dist_source)  # distance consistency loss

                lossG2, _ = loss_utils.gen_loss(g_real_logit2, g_fake_logit2, gan=self.opts.gan,
                                                noise_label=self.opts.flip_g)
                lossG3, _ = loss_utils.gen_loss(g_real_logit3, g_fake_logit3, gan=self.opts.gan,
                                                noise_label=self.opts.flip_g)
                lossG4, _ = loss_utils.gen_loss(g_real_logit4, g_fake_logit4, gan=self.opts.gan,
                                                noise_label=self.opts.flip_g)
                lossG5, _ = loss_utils.gen_loss(g_real_logit5, g_fake_logit5, gan=self.opts.gan,
                                                noise_label=self.opts.flip_g)
                lossG6, _ = loss_utils.gen_loss(g_real_logit6, g_fake_logit6, gan=self.opts.gan,
                                                noise_label=self.opts.flip_g)
                lossG7, _ = loss_utils.gen_loss(g_real_logit7, g_fake_logit7, gan=self.opts.gan,
                                                noise_label=self.opts.flip_g)
                lossG8, _ = loss_utils.gen_loss(g_real_logit8, g_fake_logit8, gan=self.opts.gan,
                                                noise_label=self.opts.flip_g)
                lossG9, _ = loss_utils.gen_loss(g_real_logit9, g_fake_logit9, gan=self.opts.gan,
                                                noise_label=self.opts.flip_g)

                # _3
                if self.opts.use_log_weight:    
                    lossG = self.weight[-8] * lossG2 + self.weight[-7] * lossG3 + self.weight[-6] * lossG4 + self.weight[-5] * lossG5 + self.weight[-4] * lossG6 + self.weight[-3] * lossG7 + self.weight[-2] * lossG8 + self.weight[-1] * lossG9 + 0.001 * rel_loss
                else:
                    lossG = 0.001 * ( lossG2 + lossG3 + lossG4) + 0.1 * (lossG5 + lossG6) + lossG7 + lossG8 + lossG9 + 0.001 * rel_loss
                # if not torch.any(torch.isnan(lossG)):
                lossG.backward()
                self.optimizerG.step()
                self.accumulate(self.G_ema, self.G)

                real_point1 = gt[-1].squeeze(0).cuda().data.cpu().numpy()

                fakepoints1 = g_fake_pred[0]
                fakepoints2 = g_fake_pred[1]
                fakepoints3 = g_fake_pred[2]
                fakepoints4 = g_fake_pred[3]
                fakepoints5 = g_fake_pred[4]
                fakepoints6 = g_fake_pred[5]
                fakepoints7 = g_fake_pred[6]
                fakepoints8 = g_fake_pred[7]
                fakepoints9 = g_fake_pred[8]

                fake_point1 = fakepoints1[0].squeeze(0).cuda().data.cpu().numpy()
                fake_point2 = fakepoints2[0].squeeze(0).cuda().data.cpu().numpy()
                fake_point3 = fakepoints3[0].squeeze(0).cuda().data.cpu().numpy()
                fake_point4 = fakepoints4[0].squeeze(0).cuda().data.cpu().numpy()
                fake_point5 = fakepoints5[0].squeeze(0).cuda().data.cpu().numpy()
                fake_point6 = fakepoints6[0].squeeze(0).cuda().data.cpu().numpy()
                fake_point7 = fakepoints7[0].squeeze(0).cuda().data.cpu().numpy()
                fake_point8 = fakepoints8[0].squeeze(0).cuda().data.cpu().numpy()
                fake_point9 = fakepoints9[0].squeeze(0).cuda().data.cpu().numpy()

                d_avg_meter.update(lossD9.item())
                g_avg_meter.update(lossG.item())

                real_acc_avg_meter.update(info['real_acc'])
                fake_acc_avg_meter.update(info['fake_acc'])

                if self.writer is not None:
                    self.writer.add_scalar("loss/d_Loss", lossD9.data, global_step)
                    self.writer.add_scalar("loss/g_Loss", lossG.data, global_step)
                    # self.writer.add_scalar("acc/real_acc", info['real_acc'], global_step)
                    # self.writer.add_scalar("acc/fake_acc", info['fake_acc'], global_step)
                    self.writer.add_scalar("lr/lr_g", self.optimizerG.param_groups[0]['lr'], global_step)
                    self.writer.add_scalar("lr/lr_d", self.optimizerD9.param_groups[0]['lr'], global_step)

                global_step += 1

            if self.scheduler_G is not None:
                self.scheduler_G.step(epoch)
            if self.scheduler_D is not None:
                self.scheduler_D.step(epoch)

            time_tick = time.time() - start_time
            self.log_string(
                "Epoch: [%2d] time: %2dm %2ds d_loss2: %.8f d_loss3: %.8f d_loss4: %.8f d_loss5: %.8f d_loss6: %.8f d_loss7: %.8f d_loss8: %.8f d_loss9: %.8f d_losslocal: %8f g_loss: %.8f" \
                % (epoch, time_tick / 60, time_tick % 60,  lossD2.data, lossD3.data, lossD4.data,
                    lossD5.data, lossD6.data, lossD7.data, lossD8.data, lossD9.data, lossDlocal.data, lossG.data))
            self.log_string("real_acc: %f  fake_acc: %f" % (real_acc_avg_meter.avg, fake_acc_avg_meter.avg))
            self.log_string(
                "lr_g: %f  lr_d: %f" % (self.optimizerG.param_groups[0]['lr'], self.optimizerD9.param_groups[0]['lr']))
            print("step_d:%d step_g:%d" % (step_d, step_g))
            # if self.scheduler_G is not None and self.scheduler_D is not None:
            #     print("lr_g: %f  lr_d: %f"%(self.scheduler_G.get_lr()[0],self.scheduler_D.get_lr()[0]))

            if epoch % self.opts.snapshot == 0:
                self.save(self.opts.log_dir, epoch)

            # if not os.path.exists(self.opts.save_path):
            #     os.makedirs(self.opts.save_path)
            self.create_dirs(self.opts.save_path)
            if not os.path.exists(os.path.join(self.opts.save_path, 'txt')):
                os.mkdir(os.path.join(self.opts.save_path, 'txt'))
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'gt1.txt'), real_point1, fmt="%f;%f;%f")

            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation1.txt'), fake_point1, fmt="%f;%f;%f")
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation2.txt'), fake_point2, fmt="%f;%f;%f")
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation3.txt'), fake_point3, fmt="%f;%f;%f")
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation4.txt'), fake_point4, fmt="%f;%f;%f")
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation5.txt'), fake_point5, fmt="%f;%f;%f")
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation6.txt'), fake_point6, fmt="%f;%f;%f")
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation7.txt'), fake_point7, fmt="%f;%f;%f")
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation8.txt'), fake_point8, fmt="%f;%f;%f")
            np.savetxt(osp.join(self.opts.save_path, 'txt', str(epoch) + 'generation9.txt'), fake_point9, fmt="%f;%f;%f")
            
            pcds = [gt[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints1[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints2[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints3[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints4[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints5[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints6[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints7[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints8[0:16].squeeze(0).cuda().data.cpu().numpy(),
                    fakepoints9[0:16].squeeze(0).cuda().data.cpu().numpy()]
            for k in range(len(up_points)):
                up_points[k] = up_points[k].permute(0, 2, 1)[0:16]
                up_points[k] = up_points[k].detach().cpu().numpy()
                up_points[k] = normalize_point_cloud(up_points[k])
            for i,pcd in enumerate(pcds):
                pcds[i] = normalize_point_cloud(pcd)
            if not os.path.exists(os.path.join(self.opts.save_path, 'plot')):
                os.mkdir(os.path.join(self.opts.save_path, 'plot'))
            # plot_multi_scale_pcd(os.path.join(self.opts.save_path, f'plot/ep_{epoch}.png'),pcds)
            # plot_multi_scale_up_pcd(os.path.join(self.opts.save_path, f'plot/ep_{epoch}.png'),pcds[1:],up_points)
            plot_multi_scale_up_diff_pcd(os.path.join(self.opts.save_path, f'plot/ep_{epoch}.png'),pcds[1:],up_points)

        self.save(self.opts.log_dir, epoch)
        print('finish')
        self.LOG_FOUT.close()

    def evaluate_gen(self):
        self.build_model()
        self.load(self.opts.checkpoint_dir)
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        # self.train_dataset = H5DataLoader(self.opts, augment=self.opts.augment)  # (6778,2048,3)
        # self.dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.opts.bs, shuffle=True,
        #                                               num_workers=int(self.opts.workers), drop_last=True,
        #                                               pin_memory=True)
        dataloader, x_train_var = get_data_loaders(self.opts)
        self.dataloader = dataloader['test_loader']
        # loader = get_test_loader(args)
        save_dir = os.path.join(self.opts.save_path,f"{self.opts.choice}/{current_time}")
        plot_dir = os.path.join(save_dir,'plot')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        all_sample = []
        all_ref = []
        for data in self.dataloader:
            idx_b, te_pc = data['idx'], data['test_points']
            te_pc = te_pc.cuda() if self.opts.gpu is None else te_pc.cuda(self.opts.gpu)
            B, N = te_pc.size(0), te_pc.size(1)
            z = self.noise_generator(bs=B)
            # out_pc = self.G(z)  # [bs,3,2048]
            # out_pcs, up_points = self.G_ema(z)  # [bs,3,2048]
            out_pcs, up_points = self.G(z)  # [bs,3,2048]
            out_pc = out_pcs[-1]
            # _, out_pc = model.sample(B, N)
            out_pc = self.normalize_point_cloud_tensor(out_pc)
            te_pc = self.normalize_point_cloud_tensor(te_pc)
            # denormalize
            # m, s = data['mean'].float(), data['std'].float()
            # m = m.cuda() if self.opts.gpu is None else m.cuda(self.opts.gpu)
            # s = s.cuda() if self.opts.gpu is None else s.cuda(self.opts.gpu)
            # out_pc = out_pc * s + m
            # te_pc = te_pc * s + m
            out_pc_plot = out_pc.cpu().numpy()
            te_pc_plot = te_pc.cpu().numpy()
            plot.plot_pcd_multi_rows(os.path.join(plot_dir,f'{idx_b[0]}_gen.png'),out_pc_plot,None)
            plot.plot_pcd_multi_rows(os.path.join(plot_dir,f'{idx_b[0]}_test.png'),te_pc_plot,None)

            all_sample.append(out_pc)
            all_ref.append(te_pc)

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)
        print("Generation sample size:%s reference size: %s"
            % (sample_pcs.size(), ref_pcs.size()))

        # Save the generative output
        # save_dir = os.path.dirname(self.opts.save_path)
        # save_dir = os.path.join(self.opts.save_path,f"{self.opts.choice}/{current_time}")
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        np.save(os.path.join(save_dir, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy())
        np.save(os.path.join(save_dir, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())

        # Compute metrics
        results = compute_all_metrics(sample_pcs, ref_pcs, self.opts.bs, accelerated_cd=True, compute_nna=True, compute_jsd=True)
        results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in results.items()}
        pprint(results)

        # sample_pcl_npy = sample_pcs.cpu().detach().numpy()
        # ref_pcl_npy = ref_pcs.cpu().detach().numpy()
        # jsd = JSD(sample_pcl_npy, ref_pcl_npy)
        # print("JSD:%s" % jsd)
        # results['JSD'] = jsd
        results['ckpt_dir'] = self.opts.checkpoint_dir
        results['model'] = self.opts.pretrain_model_G
        self.save_json(save_dir, results)

    def normalize_point_cloud_tensor(self, inputs):
        """
        input: pc [N, P, 3]
        output: pc, centroid, furthest_distance
        """
        #print("shape",input.shape)
        C = inputs.shape[-1]
        pc = inputs[:,:,:3]
        if C > 3:
            nor = inputs[:,:,3:]

        centroid = torch.mean(pc, axis=1, keepdim=True)
        pc = inputs[:,:,:3] - centroid
        furthest_distance = torch.amax(
            torch.sqrt(torch.sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
        pc = pc / furthest_distance
        if C > 3:
            return torch.cat([pc,nor],axis=-1)
        else:
            return pc

    def save_json(self, dir, res):
        if not os.path.exists(dir):
            os.mkdir(dir)
        # with open(os.path.join(dir,"res.json"), 'w') as file:
        #     json.dump(res, file)
        with open(os.path.join(dir,"res.json"), 'w') as file:
            for key, value in res.items():
                line = json.dumps({key: value}) + '\n'
                file.write(line)
        file.close()
    
    def create_dirs(self, base):
        if not os.path.exists(base):
            os.makedirs(base)
            os.mkdir(os.path.join(base, 'plot'))
            os.mkdir(os.path.join(base, 'txt'))
            os.mkdir(os.path.join(base, 'ckpt'))

    def set_grad(self, Module_num):
        Module = [self.G, self.D2, self.D3, self.D4, self.D5, self.D6, self.D7, self.D8, self.D9]
        for i,module in enumerate(Module):
            if i == Module_num:
                requires_grad(module, True)
            else:
                requires_grad(module, False)

    def get_D_weight(self):
        self.weight = []
        for i in range(1, 9):
            x = i/8
            # weight = np.log(x*(math.e -1) + 1)
            weight = np.sin(x*(np.pi/2))
            self.weight.append(weight)

    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str + '\n')
        self.LOG_FOUT.flush()
        print(out_str)

    def set_logger(self):
        self.logger = logging.getLogger("CLS")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(os.path.join(self.opts.log_dir, "log_%s.txt" % self.opts.phase))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def load_not_match_dict(self, module, ckpt_dict):
        state_dict = module.state_dict()
        model_dict = {}
        for k,v in ckpt_dict.items():
            for i,j in state_dict.items():
                _,m = i.split(".", 1)
                if m == k:
                    model_dict[i] = v
        state_dict.update(model_dict)
        module.load_state_dict(state_dict)
        return module


    def load(self, checkpoint_dir):
        if self.opts.pretrain_model_G is None and self.opts.pretrain_model_D is None:
            print('################ new training ################')
            return False, 1

        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        # ----------------- load G -------------------
        if not self.opts.pretrain_model_G is None:
            resume_file_G = os.path.join(checkpoint_dir, self.opts.pretrain_model_G)
            flag_G = os.path.isfile(resume_file_G),
            if flag_G == False:
                print('G--> Error: no checkpoint directory found!')
                exit()
            else:
                print('resume_file_G------>: {}'.format(resume_file_G))
                checkpoint = torch.load(resume_file_G)
                # self.G = self.load_not_match_dict(self.G,checkpoint['G_model'])
                self.G.load_state_dict(checkpoint['G_model'])
                # state_dict = self.G.state_dict()
                # model_dict = {}
                # for k,v in checkpoint['G_model'].items():
                #     for i,j in state_dict.items():
                #         _,m = i.split(".", 1)
                #         if m == k:
                #             model_dict[i] = v
                # state_dict.update(model_dict)
                # self.G.load_state_dict(state_dict)
                # self.G.load_state_dict(checkpoint['G_model'])
                self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
                self.G.load_state_dict(checkpoint['G_ema_model'])
                # self.G_ema = self.load_not_match_dict(self.G_ema, checkpoint['G_ema_model'])
                G_epoch = checkpoint['G_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_G")
            exit()

        # ----------------- load D -------------------
        if not self.opts.pretrain_model_D is None:
            resume_file_D = os.path.join(checkpoint_dir, self.opts.pretrain_model_D)
            flag_D = os.path.isfile(resume_file_D)
            if flag_D == False:
                print('D--> Error: no checkpoint directory found!')
                exit()
            else:
                print('resume_file_D------>: {}'.format(resume_file_D))
                checkpoint = torch.load(resume_file_D)
                # self.D1.load_state_dict(checkpoint['D_model1'])
                self.D2 = self.load_not_match_dict(self.D2,checkpoint['D_model2'])
                self.D3 = self.load_not_match_dict(self.D3,checkpoint['D_model3'])
                self.D4 = self.load_not_match_dict(self.D4,checkpoint['D_model4'])
                self.D5 = self.load_not_match_dict(self.D5,checkpoint['D_model5'])
                self.D6 = self.load_not_match_dict(self.D6,checkpoint['D_model6'])
                self.D7 = self.load_not_match_dict(self.D7,checkpoint['D_model7'])
                self.D8 = self.load_not_match_dict(self.D8,checkpoint['D_model8'])
                self.D9 = self.load_not_match_dict(self.D9,checkpoint['D_model9'])
                # self.D2.load_state_dict(checkpoint['D_model2'])
                # self.D3.load_state_dict(checkpoint['D_model3'])
                # self.D4.load_state_dict(checkpoint['D_model4'])
                # self.D5.load_state_dict(checkpoint['D_model5'])
                # self.D6.load_state_dict(checkpoint['D_model6'])
                # self.D7.load_state_dict(checkpoint['D_model7'])
                # self.D8.load_state_dict(checkpoint['D_model8'])
                # self.D9.load_state_dict(checkpoint['D_model9'])
                # self.optimizerD1.load_state_dict(checkpoint['D_optimizer1'])
                self.optimizerD2.load_state_dict(checkpoint['D_optimizer2'])
                self.optimizerD3.load_state_dict(checkpoint['D_optimizer3'])
                self.optimizerD4.load_state_dict(checkpoint['D_optimizer4'])
                self.optimizerD5.load_state_dict(checkpoint['D_optimizer5'])
                self.optimizerD6.load_state_dict(checkpoint['D_optimizer6'])
                self.optimizerD7.load_state_dict(checkpoint['D_optimizer7'])
                self.optimizerD8.load_state_dict(checkpoint['D_optimizer8'])
                self.optimizerD9.load_state_dict(checkpoint['D_optimizer9'])

                D_epoch = checkpoint['D_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_D")
            exit()

        print(" [*] Success to load model --> {} & {}".format(self.opts.pretrain_model_G, self.opts.pretrain_model_D))
        return True, G_epoch

    def save(self, checkpoint_dir, index_epoch):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_name = str(index_epoch) + '_' + self.opts.cates[0]
        path_save_G = os.path.join(checkpoint_dir, save_name + '_G.pth')
        path_save_D = os.path.join(checkpoint_dir, save_name + '_D.pth')

        print('Save Path for G: {}'.format(path_save_G))
        print('Save Path for D: {}'.format(path_save_D))

        torch.save({
            'G_ema_model':self.G_ema.module.state_dict() if self.multi_gpu else self.G.state_dict(),
            'G_model': self.G.module.state_dict() if self.multi_gpu else self.G.state_dict(),
            'G_optimizer': self.optimizerG.state_dict(),
            'G_epoch': index_epoch,
        }, path_save_G)
        torch.save({
            # 'D_model1': self.D1.module.state_dict() if self.multi_gpu else self.D1.state_dict(),
            'D_model2': self.D2.module.state_dict() if self.multi_gpu else self.D2.state_dict(),
            'D_model3': self.D3.module.state_dict() if self.multi_gpu else self.D3.state_dict(),
            'D_model4': self.D4.module.state_dict() if self.multi_gpu else self.D4.state_dict(),
            'D_model5': self.D5.module.state_dict() if self.multi_gpu else self.D5.state_dict(),
            'D_model6': self.D6.module.state_dict() if self.multi_gpu else self.D6.state_dict(),
            'D_model7': self.D7.module.state_dict() if self.multi_gpu else self.D7.state_dict(),
            'D_model8': self.D8.module.state_dict() if self.multi_gpu else self.D8.state_dict(),
            'D_model9': self.D9.module.state_dict() if self.multi_gpu else self.D9.state_dict(),
            # 'D_optimizer1': self.optimizerD1.state_dict(),
            'D_optimizer2': self.optimizerD2.state_dict(),
            'D_optimizer3': self.optimizerD3.state_dict(),
            'D_optimizer4': self.optimizerD4.state_dict(),
            'D_optimizer5': self.optimizerD5.state_dict(),
            'D_optimizer6': self.optimizerD6.state_dict(),
            'D_optimizer7': self.optimizerD7.state_dict(),
            'D_optimizer8': self.optimizerD8.state_dict(),
            'D_optimizer9': self.optimizerD9.state_dict(),
            'D_epoch': index_epoch,
        }, path_save_D)

        # torch.save(G, os.path.join(opt.outd, opt.outm, f'G_nch-{opt.nch}_epoch-{epoch}.pth'))
        # torch.save(D, os.path.join(opt.outd, opt.outm, f'D_nch-{opt.nch}_epoch-{epoch}.pth'))
        # torch.save(Gs, os.path.join(opt.outd, opt.outm, f'Gs_nch-{opt.nch}_epoch-{epoch}.pth'))



