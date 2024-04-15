# from datasets import get_datasets, synsetid_to_cate
# from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import emd_cd
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
# from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn

from Generation_snow_style_S_adapt_stylegan.config_2_const import opts
from Generation_snow_style_S_adapt_stylegan.model_PS_2_const_dis import Model



print("Starting...")

def main(args):
    print("Creating Model...")
    model = Model(args)
    print("Model created successfully.")

    def _transform_(m):
        return nn.DataParallel(m)

    print("Resume Path:%s" % args.checkpoint_dir)

    with torch.no_grad():
        if args.evaluate_recon:
            # Evaluate reconstruction
            # evaluate_recon(model, args)
            pass
        else:
            model.evaluate_gen()


if __name__ == '__main__':
    with torch.cuda.device(5):
        opts.use_local_D = False
        args = opts
        args.bs = 48
        args.batch_size = 48
        args.choice = "chair"
        args.cates = ['chair']
        args.save_path = "results/test"
        args.checkpoint_dir = "/media/data/zhiqiang/pointstylegan2/ckpts/chair/20231128-1022"
        args.pretrain_model_G = "5500_chair_G.pth"
        args.pretrain_model_D = "5500_chair_D.pth"
        main(args)
