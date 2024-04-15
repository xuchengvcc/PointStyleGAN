#!/usr/bin/env python  
#-*- coding:utf-8 _*-

import argparse
import os

def str2bool(x):
    return x.lower() in ('true')


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def check_args(args):
    if args.model_dir is None:
        print('please create model dir')
        exit()
    if args.network is None:
        print('please select model!!!')
        exit()
    check_folder(args.checkpoint_dir)                                   # --checkpoint_dir
    check_folder(os.path.join(args.checkpoint_dir, args.model_dir))     # --chekcpoint_dir + model_dir

    try: # --epoch
        assert args.max_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    try: # --batch_size
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

#root = os.getcwd()[:5]
# define the H5 data folder
# data_root_h5 = "/home/lzq/Generation/H5/train_h5"
data_root_h5 = "/media/data/zhiqiang/dataset/Generation/H5/train_h5/shapenet_part"
pretain_model = ""

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test ?')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--bs', type=int, default=16, help='input batch size [default: 64]')
parser.add_argument('--np', type=int, default=64, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--nk',type=int, default=20,help = 'number of the knn graph point')
parser.add_argument('--num_points', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--start_epoch', type=int, default=1, help='number of epochs to start')
parser.add_argument('--dataset_num', type=int, default=3000, help='nums of samples in the dataset')
parser.add_argument("--kl_wt", type=int, default=100)
parser.add_argument("--feat_const_batch", type=int, default=16) #60
parser.add_argument('--lr_decay', action='store_true', help='use offset')
parser.add_argument('--nv', type=float, default=1.0, help='value of noise') #1.0/0.2

parser.add_argument('--gan', default='gan', help='[ls,wgan,hinge]')
parser.add_argument('--flip_d', default=True,action='store_true',help='use offset')
parser.add_argument('--flip_g', default=True,action='store_true',help='use offset')
parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--lr_decay_feq', type=int, default=200, help='use offset')
parser.add_argument('--log_dir', default='/media/data/zhiqiang/pointstylegan2/ckpts', help='log_dir')
parser.add_argument('--tensorboard_path', default='/media/data/zhiqiang/output/results/run/stylegan/chair2/2_3_dis2_loss4_2', help='directory to save generated point clouds')
parser.add_argument('--save_path', default='/home/zhiqiang/home/ubuntu/xiangmu/spgan/results/train_save/new/stylegan/chair2/32/4', help='directory to save generated point clouds')
parser.add_argument('--save_dir', default='/home/zhiqiang/home/ubuntu/xiangmu/spgan/results/log/new/stylegan/chair2/32/1', help='log_dir')
parser.add_argument('--test_save_path', default='/home/zhiqiang/home/ubuntu/xiangmu/spgan/results/test/', help='directory to save test point clouds sample')

parser.add_argument('--modelG', default='model_kl', help='directory to save generated point clouds')


parser.add_argument('--lr_g', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--lr_d', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--nz', type=int, default=512, help='dimensional of noise')
###stylegam_point
parser.add_argument('--n_mlp', type=int, default=4, help='dimensional of noise')
parser.add_argument('--lr_mlp', type=float, default=0.01, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--n_latent', type=int, default=18, help='dimensional of noise')
parser.add_argument('--iter_G', type=int, default=2, help='dimensional of noise')
parser.add_argument('--channel_multiplier', type=int, default=2, help='dimensional of noise')




parser.add_argument('--scale', type=float, default=1.0, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--neg', type=float, default=0.01, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--neg2', type=float, default=0.01, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--save', action='store_true', help='use offset')
parser.add_argument('--augment', type=str2bool, default=False, help='use offset')
parser.add_argument('--off',action='store_true', help='use offset')
parser.add_argument('--part', action='store_true', help='use offset')
parser.add_argument('--part_more', action='store_true', help='use offset')
parser.add_argument('--moving', action='store_true', help='use offset')
parser.add_argument('--max_epoch', type=int, default=6000, help='number of epochs to train for')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--debug', type=bool, default = True,  help='print log')
parser.add_argument('--data_root', type=str,default=data_root_h5, help='data root [default: xxx]')
parser.add_argument('--log_info', default='log_info.txt', help='log_info txt')
parser.add_argument('--model_dir', default='PDGN_v1', help='model dir [default: None, must input]')
parser.add_argument('--checkpoint_dir', default='checkpoint', help='Checkpoint dir [default: checkpoint]')
parser.add_argument('--snapshot', type=int, default=50, help='how many epochs to save model')
parser.add_argument('--choice', default='Chair', help='choice class')
parser.add_argument('--network', default='PDGN_v1', help='which network model to be used')
parser.add_argument('--savename',default = 'PDGN_v1',help='the generate data name')
parser.add_argument('--pretrain_model_D', default='1050_chair_D.pth', help='use the pretrain model D')
parser.add_argument('--pretrain_model_G', default='1050_chair_G.pth', help='use the pretrain model G')
parser.add_argument('--pretmax', type=str2bool, default=True, help='softmax for bilaterl interpolation')
parser.add_argument('--dataset', default='shapenet', help='choice dataset [shapenet, modelnet10, modelnet40]')
parser.add_argument('--restore', action='store_true')
parser.add_argument('--min', action='store_true')
parser.add_argument('--atx', action='store_true')
parser.add_argument('--gctn', action='store_true')
parser.add_argument('--n_mix', action='store_true')
parser.add_argument('--w_mix', action='store_true')
parser.add_argument('--trunc', action='store_true')
parser.add_argument('--use_sgd', action='store_true')
parser.add_argument('--n_rand', action='store_true')
parser.add_argument('--sn', action='store_true')
parser.add_argument('--z_norm', action='store_true')
parser.add_argument('--bal', action='store_true')
parser.add_argument('--bal_para', type=float, default=0.15, help='value of noise')
parser.add_argument('--bal_epoch', type=int, default=30, help='how many epochs to save model')
parser.add_argument('--norm',default = 'IN',help='"BN","IN","PN"')
parser.add_argument('--d_iter', type=int, default=1, help='how many epochs to save model')
parser.add_argument('--g_iter', type=int, default=1, help='how many epochs to save model')
parser.add_argument('--no_global', action='store_true')
parser.add_argument('--dp', action='store_true')
parser.add_argument('--use_noise', action='store_true')
parser.add_argument('--noise_label', action='store_true',help='use offset')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--inst_noise', action='store_true')
parser.add_argument('--small_d',action='store_true', help='use offset')
parser.add_argument('--cut_d',action='store_true', help='use offset')
parser.add_argument('--keep_idx',action='store_true', help='use offset')
parser.add_argument('--cat',action='store_true', help='use offset')
parser.add_argument('--gat',action='store_true', help='use offset')
parser.add_argument('--same_head',action='store_true', help='use offset')
parser.add_argument('--use_head',action='store_true', help='use offset')
parser.add_argument('--lr_decay_g',action='store_true', help='use offset')
parser.add_argument('--lr_decay_d', action='store_true', help='use offset')
parser.add_argument('--attn', action='store_true')

parser.add_argument('--ema_rate', type=float, default=0.999, help='value of ema_rate')
parser.add_argument('--BN', action='store_true', help='use BatchNorm in G and D')
parser.add_argument('--WS', action='store_true', help='use WeightScale in G and D')
parser.add_argument('--eql', action='store_true')
parser.add_argument('--PN', action='store_true', help='use PixelNorm in G')
parser.add_argument('--res', action='store_true', help='use PixelNorm in G')
parser.add_argument('--con', action='store_true')
parser.add_argument('--cls',type=int, default=2,help = 'number of the knn graph point')

parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')

# Network argumentssoftmax`
parser.add_argument('--epochs', type=int, default=2000, help='Integer value for epochs.')
parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
parser.add_argument('--DEGREE', type=int, default=[1, 2, 2, 2, 2, 2, 64], nargs='+',
                          help='Upsample degrees for generator.')
parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+',
                          help='Features for generator.')
parser.add_argument('--D_FEAT', type=int, default=[3, 64, 128, 256, 512, 1024], nargs='+',
                          help='Features for discriminator.')

parser.add_argument('--lr_t', type=float, default=1e-4, help='Float value for learning rate.')
parser.add_argument('--lr_p', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')

###CRN_dataset
parser.add_argument('--split', type=str, default='train',help='NOTE: train if pretrain and generate_fpd_stats; test otherwise')
parser.add_argument('--class_choice', type=str, default='chair',help='plane|cabinet|car|chair|lamp|couch|table|watercraft')

parser.add_argument('--evaluate_recon', type=bool, default=False, help='Set the testing Mod')
parser.add_argument('--gpu',type=str,default=None,help='select gpu')

# ShapeNet Dataset
parser.add_argument('--data_dir', type=str, default="/dataset/shapenetcore/ShapeNetCore_v2_15k.hdf5", help="Path to the training data")
parser.add_argument('--cates', type=str, nargs='+', default=["airplane"],help="Categories to be trained (useful only if 'shapenet' is selected)")
parser.add_argument('--batch_size', type=int, default=64,help='Batch size (of datasets) for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument("--tr_max_sample_points", type=int, default=2048,help='Max number of sampled points (train)')
parser.add_argument("--te_max_sample_points", type=int, default=2048,help='Max number of sampled points (test)')
parser.add_argument('--dataset_type', type=str, default="shapenet1024",help="Dataset types.", choices=['shapenet1024', 'shapenet2048'])
parser.add_argument('--dataset_scale', type=float, default=1.,help='Scale of the dataset (x,y,z * scale = real output, default=1).')
parser.add_argument('--normalize_per_shape', action='store_true',help='Whether to perform normalization per shape.')
parser.add_argument('--normalize_std_per_axis', action='store_true',help='Whether to perform normalization per axis.')

parser.add_argument('--use_local_D', type=bool, default=True, help='use local patch discriminator or not')
parser.add_argument('--use_log_weight', type=bool, default=True, help='use local patch discriminator or not')

opts = check_args(parser.parse_args())


