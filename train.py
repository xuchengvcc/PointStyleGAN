#!/usr/bin/env python  
#-*- coding:utf-8 _*-  

import os
import torch
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime
from Generation_snow_style_S_adapt_stylegan.model_PS_2_const_dis import Model
from Generation_snow_style_S_adapt_stylegan.config_2_const import opts

if __name__ == '__main__':
    with torch.cuda.device(0):
        opts.use_local_D = False
        opts.use_log_weight = True
        opts.choice = "chair"
        opts.cates = ['chair']
        opts.bs = 96
        opts.batch_size = opts.bs
        opts.num_workers = opts.workers
        opts.save_path = "results/Generation/chair/11-28"
        if opts.phase == "train":
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            opts.log_dir = os.path.join(opts.log_dir, opts.choice, current_time)
            if not os.path.exists(opts.log_dir):
                os.makedirs(opts.log_dir)

        print('checkpoints:', opts.log_dir)

        model = Model(opts)
        model.train()


