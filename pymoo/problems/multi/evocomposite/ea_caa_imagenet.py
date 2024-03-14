import argparse
import csv
import os
import random
import time
import builtins
from typing import Dict, List
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from composite_adv.attacks import *
from composite_adv.utilities import make_dataloader, EvalModel
from math import pi
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
dataset_normalizer = {'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
                      'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}


def list_type(s):
    return tuple(sorted(map(int, s.split(','))))


parser = argparse.ArgumentParser(
    description='Model Robustness Evaluation')
# parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
#                     help='attack names')
# parser.add_argument('attacks', type=str,default="AutoLinfAttack(model, 'cifar', bound=8/255)", nargs='+',
#                     help='attack names')
parser.add_argument('--arch', type=str, default='resnet50',
                    help='model architecture')
parser.add_argument('--checkpoint', type=str, default='/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/imagenet_gat_resnet50_cpu.pt',
                    help='checkpoint path')
parser.add_argument('--input-normalized', default=True, action='store_true',
                    help='model is trained for normalized data')
parser.add_argument('--dataset', type=str, default='imagenet',
                    help='dataset name')
parser.add_argument('--dataset-path', type=str, default='/mnt/jfs/wangdonghua/dataset/ImageNet/val',
                    help='path to datasets directory')
parser.add_argument('--batch-size', type=int, default=32,
                    help='number of examples/minibatch')
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


def main():
    # settings
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

    vns(args.gpu, args)

def vns(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))

    from composite_adv.utilities import make_model
    base_model = make_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # Uncomment the following if you want to load their checkpoint to finetuning
    # from composite_adv.utilities import make_madry_model, make_trades_model
    # base_model = make_madry_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # base_model = make_trades_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # base_model = make_pat_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # base_model = make_fast_at_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)

    model = EvalModel(base_model, normalize_param=dataset_normalizer[args.dataset],
                      input_normalized=args.input_normalized)
    # x = [0,1,1]
    # # y = main_worker(args, x, model)
    # # print(y)
    # search_space = [0,1,2]
    # max_length = 6
    # for i in range(3, max_length):
    #     for j in range(len(x)):
    #         for k in range(3):
    #             x[j] = search_space[k]
    #
    #             y = main_worker(args, x, model)
    #             print(x)
    #             print(y)
    dimension = 3
    x_min = 0
    x_max = 3
    ns = NeighborSearch(dimension, x_min, x_max, model, args)
    print('Initialization succeed.')
    solution = ns.evolve()
    print('The searched attack:')
    print(solution)

class NeighborSearch():
    def __init__(self, dim, lower_bound, upper_bound, model, args):
        self.dim = dim  # 设计变量维度
        self.x_bound_lower = lower_bound
        self.x_bound_upper = upper_bound    # 设计变量上界，注意取不到该数值
        self.net = model
        # self.x = np.zeros((1, self.dim))
        self.x = np.array([0,2,0])
        self.args = args
        self.pg = self.x
        self.model = model
        fitness = self.calculate_fitness(self.x)
        self.pg_fitness = fitness


    def calculate_fitness(self, x):
        print(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        fitness = np.zeros([x.shape[0]])
        for j in range(x.shape[0]):
            fitness[j] = main_worker(self.args, self.model, x[j][:].astype(int))
        return fitness

    def neighborhood(self, x, location):
        neighbor = np.zeros([2, self.dim])
        k = 0
        # for i in range(self.dim):
        for j in np.arange(self.x_bound_lower, self.x_bound_upper):
            # if np.isin(j, x):
            if j == x[location]:
                continue
            neighbor_x = x.copy()
            neighbor_x[location] = j
            # neighbor_x = np.sort(neighbor_x)
            # if neighbor_x.ndim == 1:
            #     neighbor_x = neighbor_x[np.newaxis, :]
            neighbor[k][:] = neighbor_x
            k = k + 1
        # print(neighbor)
        return neighbor

    def evolve(self):
        iteration_best_fitness = self.pg_fitness
        flag = 1   # 用来指示目标函数是否有改进
        step = 0
        indicator = 0
        count = 0
        lenth = self.dim
        Fitness = []
        # while flag == 1:
        while self.dim < 8:
            while count<100:
                flag = 0
                indicator += 1
                print('Indicator = ', str(indicator))
                for i in range(self.dim):
                    count += 1
                    neighbor = self.neighborhood(self.pg, i)
                    fitness_neighbor = self.calculate_fitness(neighbor)
                    # print(neighbor)
                    temp = np.min(fitness_neighbor[:])
                    if temp < self.pg_fitness:
                        flag = 1
                        self.pg = neighbor[np.argmin(fitness_neighbor[:])]
                        self.pg_fitness = np.min(fitness_neighbor[:])
                        # step += 1
                        # print('Position： %d, Iter: %d, Best fitness: %.5f' % (i, step, self.pg_fitness))
                        # iteration_best_fitness = np.append(iteration_best_fitness, self.pg_fitness)
                        # break
                    print(self.pg_fitness)
                    Fitness.append(self.pg_fitness)
                    iteration_best_fitness = np.append(iteration_best_fitness, self.pg_fitness)
                    step += 1
                    # print('Position: %d, Iter: %d, Best fitness: %.5f' % (i, step, self.pg_fitness))
            self.dim = self.dim + 1
            a = np.random.randint(0,3,dtype='int')*np.ones(1)
            self.pg = np.concatenate((self.pg, a), axis=0)
            count = 0
        plt.plot(Fitness)
        plt.savefig('fitness.png')
        return self.pg



def main_worker(args, model, x):
    attacks = []
    # for attack_name in attack_names:
    # x = [2,0,1, 2, 1,0] 51.7
    # x = [0, 1, 2]  44.8
    # x = [0, 2, 1]  37.9
    # x = [1, 2, 0] 48.3
    # x = [1, 0, 2]  48.3
    # x = [2, 0, 1] 41.4
    # x = [2, 1, 0]  44.8
    # print(tuple(x))
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

    fitness = evaluate(model, test_loader, attack_names, attacks, args)
    return fitness
    # eval_test(model, test_loader,args)

def evaluate(model, val_loader, attack_names, attacks, args):
    model.eval()

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_ori_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_time_used: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    Dist_norm = 0
    count = 0
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
            batch_tic = time.perf_counter()
            adv_inputs = attack(inputs, labels)

            dist = (adv_inputs - inputs)

            # L2
            dist = dist.view(inputs.shape[0], -1)
            dist_norm = torch.norm(dist, dim=1, keepdim=True)
            dist_norm = torch.sum(dist_norm)
            # MSE
            # dist_norm = torch.sum((dist) ** 2)
            Dist_norm = Dist_norm + dist_norm
            ae = adv_inputs[0,:,:,:]
            images = ae.squeeze(0)
            # print(images.shape)
            # plt.axis('off')
            # plt.imshow(images.permute(1, 2, 0).cpu().detach().numpy())
            # plt.savefig('/mnt/jfs/sunjialiang/AAAD/noise_visualization/CAA/test1.png', bbox_inches='tight', pad_inches=0.02)

            with torch.no_grad():
                ori_logits = model(inputs)
                adv_logits = model(adv_inputs)
            batch_ori_correct = (ori_logits.argmax(1) == labels).detach()
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            batch_attack_success_rate = 1.0 - batch_correct[batch_ori_correct].float().mean().item()
            batch_toc = time.perf_counter()
            time_used = torch.tensor(batch_toc - batch_tic)
            # print(dist_norm)
            # print(f'ATTACK {attack_name}',
            #       f'accuracy = {batch_accuracy * 100:.1f}',
            #       f'attack_success_rate = {batch_attack_success_rate * 100:.1f}',
            #       f'time_usage = {time_used:0.2f} s',
            #       f'l2_norm = {dist_norm:.1f} ',
            #       sep='\t')
            batches_ori_correct[attack_name].append(batch_ori_correct)
            batches_correct[attack_name].append(batch_correct)
            batches_time_used[attack_name].append(time_used)
        # print(count)
        if count>0:
            break
    return batch_accuracy * 100

    # with open(args.output, 'a+') as out_file:
    #     out_csv = csv.writer(out_file)
    #     out_csv.writerow([args.message])
    #     out_csv.writerow(['attack_setting'] + attack_names)
    #     out_csv.writerow(['accuracies'] + accuracies)
    #     out_csv.writerow(['attack_success_rates'] + attack_success_rates)
    #     out_csv.writerow(['time_usage'] + total_time_used)
    #     out_csv.writerow(['batch_size', args.batch_size])
    #     out_csv.writerow(['num_batches', args.num_batches])
    #     out_csv.writerow([''])


if __name__ == '__main__':
    main()