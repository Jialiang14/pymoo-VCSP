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


class ZDT(Problem):

    def __init__(self, n_var=7, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=4, vtype=float, **kwargs)


class ZDT1(ZDT):

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
        # print('产生个体',x)
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


class ZDT2(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.power(x, 2)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - anp.power((f1 * 1.0 / g), 2))

        print('f1',f1)

        out["F"] = anp.column_stack([f1, f2])


class ZDT3(ZDT):

    def _calc_pareto_front(self, n_points=100, flatten=True):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        pf = []

        for r in regions:
            x1 = np.linspace(r[0], r[1], int(n_points / len(regions)))
            x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
            pf.append(np.array([x1, x2]).T)

        if not flatten:
            pf = np.concatenate([pf[None,...] for pf in pf])
        else:
            pf = np.row_stack(pf)

        return pf

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - anp.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * anp.sin(10 * anp.pi * f1))

        out["F"] = anp.column_stack([f1, f2])


class ZDT4(ZDT):
    def __init__(self, n_var=10):
        super().__init__(n_var)
        self.xl = -5 * np.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 5 * np.ones(self.n_var)
        self.xu[0] = 1.0
        self.func = self._evaluate

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1.0
        g += 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[:, i] * x[:, i] - 10.0 * anp.cos(4.0 * anp.pi * x[:, i])
        h = 1.0 - anp.sqrt(f1 / g)
        f2 = g * h

        out["F"] = anp.column_stack([f1, f2])


class ZDT5(ZDT):

    def __init__(self, m=11, n=5, normalize=True, **kwargs):
        self.m = m
        self.n = n
        self.normalize = normalize
        super().__init__(n_var=(30 + n * (m - 1)), **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = 1 + np.linspace(0, 1, n_pareto_points) * 30
        pf = np.column_stack([x, (self.m-1) / x])
        if self.normalize:
            pf = normalize(pf)
        return pf

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(float)

        _x = [x[:, :30]]
        for i in range(self.m - 1):
            _x.append(x[:, 30 + i * self.n: 30 + (i + 1) * self.n])

        u = anp.column_stack([x_i.sum(axis=1) for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[:, 1:].sum(axis=1)

        f1 = 1 + u[:, 0]
        f2 = g * (1 / f1)

        if self.normalize:
            f1 = normalize(f1, 1, 31)
            f2 = normalize(f2, (self.m-1) * 1/31, (self.m-1))

        out["F"] = anp.column_stack([f1, f2])


class ZDT6(ZDT):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0.2807753191, 1, n_pareto_points)
        return np.array([x, 1 - np.power(x, 2)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1 - anp.exp(-4 * x[:, 0]) * anp.power(anp.sin(6 * anp.pi * x[:, 0]), 6)
        g = 1 + 9.0 * anp.power(anp.sum(x[:, 1:], axis=1) / (self.n_var - 1.0), 0.25)
        f2 = g * (1 - anp.power(f1 / g, 2))

        out["F"] = anp.column_stack([f1, f2])
