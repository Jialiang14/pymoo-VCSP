'''
求解问题部分
'''
import os
import argparse
import csv
import random
import time
import builtins
from typing import Dict, List
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import scipy.io as scio
from pymoo.problems.multi.evocomposite.composite_adv.attacks import *
from pymoo.problems.multi.evocomposite.composite_adv.utilities import make_dataloader, EvalModel
from math import pi
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from pymoo.problems.multi.models import *
import numpy as np


import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem
from pymoo.util.normalization import normalize


class VCSP(Problem):

    def __init__(self, n_var=7, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=4, vtype=float, **kwargs)

# 函数变量维度（目标维度不一致的自行编写目标函数）
class VCSP1(VCSP):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        # print('产生个体',x)
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)
        f2 = g * (1 - anp.power((f1 / g), 0.5))

        out["F"] = anp.column_stack([f1, f2])

    def _evaluate(self, x, out, *args, **kwargs):
        print('产生个体',x)
        # settings
        warnings.filterwarnings('ignore')
        dataset_normalizer = {'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
                              'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
        parser = argparse.ArgumentParser(
            description='Model Robustness Evaluation')
        # parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
        #                     help='attack names')
        # parser.add_argument('attacks', type=str,default="AutoLinfAttack(model, 'cifar', bound=8/255)", nargs='+',
        #                     help='attack names')
        parser.add_argument('--arch', type=str, default='wideresnet',
                            help='model architecture')
        parser.add_argument('--checkpoint', type=str,
                            default='/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/cifar_gat_finetune_trades_madry_loss_cpu.pt',
                            help='checkpoint path')
        parser.add_argument('--input-normalized', default=True, action='store_true',
                            help='model is trained for normalized data')
        parser.add_argument('--dataset', type=str, default='cifar10',
                            help='dataset name')
        parser.add_argument('--dataset-path', type=str, default='/mnt/jfs/sunjialiang/data',
                            help='path to datasets directory')
        parser.add_argument('--batch-size', type=int, default=32,
                            help='number of examples/minibatch')
        parser.add_argument('--popsize', type=int, default=20,
                            help='number of population')
        parser.add_argument('--max_gen', type=int, default=20,
                            help='number of iteration')
        parser.add_argument('--length', type=int, default=8,
                            help='length of caa')
        parser.add_argument('--num-batches', type=int, required=False,
                            help='number of batches (default entire dataset)')
        parser.add_argument('--message', type=str, default="",
                            help='csv message before result')
        parser.add_argument('--seed', type=int, default=0, help='RNG seed')
        parser.add_argument('--output', type=str, help='output CSV')
        parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                            help='number of data loading workers (default: 0)')
        parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
        parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
        parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')
        args = parser.parse_args()

        if args.seed is not None:
            # Make sure we can reproduce the testing result.
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            cudnn.deterministic = True

        if args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')


        model = VGG('VGG19')

        model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/VGG-20230326-225632/weights.pt'))
        f1 = []
        f2 = []
        for i in range(len(x)):
            indi = self.convert_to_discrete_integers(x[i,:])
            Robust_accuracy, l2_distance = self.main_worker(args, model, indi)
            l2_distance = l2_distance.cpu().detach()
            l2_distance = l2_distance.numpy()
            f1.append(Robust_accuracy)
            f2.append(l2_distance)
        f1 = np.array(f1)
        f2 = np.array(f2)
        out["F"] = anp.column_stack([f1, f2])

    def convert_to_discrete_integers(self, float_list):
        return [round(number) for number in float_list]

    def main_worker(self, args, model, x):
        attacks = []
        attack_names: List[str] = [
            "CompositeAttack(model, enabled_attack="+ str(tuple(x)) + ", order_schedule='fixed', inner_iter_num=1)"]
        for attack_name in attack_names:
            # print(attack_name)
            tmp = eval(attack_name)
            attacks.append(tmp)

            # attacks.append(tmp)
        # attacks = [tmp]
        # Send to GPU
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

        test_loader = make_dataloader(args.dataset_path, args.dataset, args.batch_size,
                                      train=False)

        fitness, l2_distance = self.evaluate_fitness(model, test_loader, attack_names, attacks, args)
        return fitness, l2_distance


    def evaluate_fitness(self, model, val_loader, attack_names, attacks, args):
        model.eval()
        Dist_norm = 0
        count = 0
        begin = time.time()
        for batch_index, (inputs, labels) in enumerate(val_loader):
            # print(f'BATCH {batch_index:05d}')

            if (
                    args.num_batches is not None and
                    batch_index >= args.num_batches
            ):
                break

            inputs = inputs.cuda()
            labels = labels.cuda()

            count = count + 1
            for attack_name, attack in zip(attack_names, attacks):
                adv_inputs = attack(inputs, labels)
                dist = (adv_inputs - inputs)
                # L2
                dist = dist.view(inputs.shape[0], -1)
                dist_norm = torch.norm(dist, dim=1, keepdim=True)
                dist_norm = torch.sum(dist_norm)
                # MSE
                # dist_norm = torch.sum((dist) ** 2)
                Dist_norm = Dist_norm + dist_norm
                # ae = adv_inputs[0,:,:,:]
                # print(images.shape)
                # plt.axis('off')
                # plt.imshow(images.permute(1, 2, 0).cpu().detach().numpy())
                # plt.savefig('/mnt/jfs/sunjialiang/AAAD/noise_visualization/CAA/test1.png', bbox_inches='tight', pad_inches=0.02)
                with torch.no_grad():
                    adv_logits = model(adv_inputs)
                batch_correct = (adv_logits.argmax(1) == labels).detach()
                batch_accuracy = batch_correct.float().mean().item()
            # print(count)
            if count>20:
                break
        time_cost = time.time() - begin
        torch.cuda.empty_cache()
        return batch_accuracy * 100,  Dist_norm/(32*21)+time_cost
