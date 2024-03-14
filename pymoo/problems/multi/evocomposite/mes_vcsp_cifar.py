import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
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
from AAA.optimizer_attack.evocomposite.composite_adv.attacks import *
from AAA.optimizer_attack.evocomposite.composite_adv.utilities import make_dataloader, EvalModel
from math import pi
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from retrain.PGD_advtrain.models import *

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
parser.add_argument('--arch', type=str, default='wideresnet',
                    help='model architecture')
parser.add_argument('--checkpoint', type=str, default='/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/cifar_gat_finetune_trades_madry_loss_cpu.pt',
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

    NSGA(args.gpu, args)

def NSGA(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))

    # from composite_adv.utilities import make_model
    # base_model = make_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # # Uncomment the following if you want to load their checkpoint to finetuning
    # # from composite_adv.utilities import make_madry_model, make_trades_model
    # # base_model = make_madry_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # # base_model = make_trades_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # # base_model = make_pat_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # # base_model = make_fast_at_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    #
    # model = EvalModel(base_model, normalize_param=dataset_normalizer[args.dataset],
    #                   input_normalized=args.input_normalized)

    args.model = 'VGG'
    if args.model == 'VGG':
        model = VGG('VGG19')
    elif args.model == 'ResNet18':
        model = ResNet18()
    elif args.model == 'GoogLeNet':
        model = GoogLeNet()
    elif args.model == 'DenseNet121':
        model = DenseNet121()
    elif args.model == 'DenseNet201':
        model = DenseNet201()
    elif args.model == 'ResNeXt29':
        model = ResNeXt29_2x64d()
    elif args.model == 'ResNeXt29L':
        model = ResNeXt29_32x4d()
    elif args.model == 'MobileNet':
        model = MobileNet()
    elif args.model == 'MobileNetV2':
        model = MobileNetV2()
    elif args.model == 'DPN26':
        model = DPN26()
    elif args.model == 'DPN92':
        model = DPN92()
    elif args.model == 'ShuffleNetG2':
        model = ShuffleNetG2()
    elif args.model == 'SENet18':
        model = SENet18()
    elif args.model == 'ShuffleNetV2':
        model = ShuffleNetV2(1)
    elif args.model == 'EfficientNetB0':
        model = EfficientNetB0()
    elif args.model == 'PNASNetA':
        model = PNASNetA()
    elif args.model == 'RegNetX':
        model = RegNetX_200MF()
    elif args.model == 'RegNetLX':
        model = RegNetX_400MF()
    elif args.model == 'PreActResNet50':
        model = PreActResNet50()

    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp_110/eval-MobileNetV2-20230531-031341/weights.pt'))
    model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/VGG-20230326-225632/weights.pt'))
    # model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/MobileNetV2-20230326-111231/weights.pt'))
    # model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/ResNet18-20230404-100449/weights.pt'))
    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/GoogLeNet-20230523-150832/weights.pt'))
    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DenseNet121-20230326-203457/weights.pt'))


    x_min = 0
    x_max = 4
    nsga = NSGASearch(x_min, x_max, model, args)
    solution = nsga.evolve()
    print('The searched attack:')
    print(solution)

class NeighborSearch():
    def __init__(self, selected_indi, lower_bound, upper_bound, model, args):
        self.dim = len(selected_indi)  # 设计变量维度
        self.x_bound_lower = lower_bound
        self.x_bound_upper = upper_bound    # 设计变量上界，注意取不到该数值
        self.net = model
        # self.x = np.zeros((1, self.dim))
        self.x = np.array(selected_indi)
        self.args = args
        self.pg = self.x
        self.model = model
        fitness = self.calculate_fitness(self.x)
        self.pg_fitness = fitness


    def calculate_fitness(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        fitness = np.zeros([x.shape[0]])
        for j in range(x.shape[0]):
            ra, l2 = main_worker(self.args, self.model, x[j][:].astype(int))
            fitness[j] = ra + 1*l2
        return fitness

    def neighborhood(self, x, location):
        neighbor = np.zeros([self.x_bound_upper-self.x_bound_lower, self.dim],dtype='int')
        k = 0
        # for i in range(self.dim):
        for j in np.arange(self.x_bound_lower, self.x_bound_upper+1):
            # if np.isin(j, x):
            if int(j) == x[location]:
                continue
            neighbor_x = x.copy()
            neighbor_x[location] = int(j)
            # neighbor_x = np.sort(neighbor_x)
            # if neighbor_x.ndim == 1:
            #     neighbor_x = neighbor_x[np.newaxis, :]
            neighbor[k][:] = neighbor_x
            k = k + 1
        # print(neighbor)
        return neighbor

    def evolve(self):
        print('Neighborhood Search')
        iteration_best_fitness = self.pg_fitness
        flag = 1   # 用来指示目标函数是否有改进
        step = 0
        indicator = 0
        Fitness = []
        length = self.dim
        # 允许变长度邻域搜索
        print(self.pg)
        while self.dim < min(length + 1, 8):
            while flag == 1:
                flag = 0
                indicator += 1
                print('Indicator = ', str(indicator))
                for i in range(self.dim):
                    neighbor = self.neighborhood(self.pg, i)
                    fitness_neighbor = self.calculate_fitness(neighbor)
                    temp = np.min(fitness_neighbor[:])
                    if temp < self.pg_fitness:
                        flag = 1
                        self.pg = neighbor[np.argmin(fitness_neighbor[:])]
                        self.pg_fitness = np.min(fitness_neighbor[:])
                    Fitness.append(self.pg_fitness)
                    iteration_best_fitness = np.append(iteration_best_fitness, self.pg_fitness)
                    step += 1
                    print(self.pg)
            self.dim = self.dim + 1
            a = np.random.randint(0, 5, dtype='int') * np.ones(1, dtype='int')
            self.pg = np.concatenate((self.pg, a), axis=0)


        # # 允许变长度邻域搜索
        # max_cal = 15
        # while count <= max_cal:
        #     indicator += 1
        #     print('Indicator = ', str(indicator))
        #     for i in range(self.dim):
        #         print(self.pg)
        #         count = count + 4
        #         neighbor = self.neighborhood(self.pg, i)
        #         print(neighbor)
        #         fitness_neighbor = self.calculate_fitness(neighbor)
        #         temp = np.min(fitness_neighbor[:])
        #         if temp < self.pg_fitness:
        #             self.pg = neighbor[np.argmin(fitness_neighbor[:])]
        #             self.pg_fitness = np.min(fitness_neighbor[:])
        #         Fitness.append(self.pg_fitness)
        #         iteration_best_fitness = np.append(iteration_best_fitness, self.pg_fitness)
        #         step += 1
        #     while count <= max_cal:
        #         self.dim = self.dim + 1
        #         a = np.random.randint(0, 5, dtype='int') * np.ones(1, dtype='int')
        #         self.pg = np.concatenate((self.pg, a), axis=0)

        # 固定长度邻域搜索
        # indicator += 1
        # print('Indicator = ', str(indicator))
        # for i in range(self.dim):
        #     print(self.pg)
        #     neighbor = self.neighborhood(self.pg, i)
        #     print(neighbor)
        #     fitness_neighbor = self.calculate_fitness(neighbor)
        #     temp = np.min(fitness_neighbor[:])
        #     if temp < self.pg_fitness:
        #         self.pg = neighbor[np.argmin(fitness_neighbor[:])]
        #         self.pg_fitness = np.min(fitness_neighbor[:])
        #     Fitness.append(self.pg_fitness)
        #     iteration_best_fitness = np.append(iteration_best_fitness, self.pg_fitness)
        return self.pg

class NSGASearch():
    def __init__(self, lower_bound, upper_bound, model, args):
        self.x_bound_lower = int(lower_bound)
        self.x_bound_upper = int(upper_bound)    # 设计变量上界，注意取不到该数值
        self.net = model
        self.x = np.array([0,2,0])
        self.args = args
        self.pg = self.x
        self.model = model
        self.max_length = args.length
        self.popsize = args.popsize

    def calculate_fitness(self, pop):
        function1 = []
        function2 = []
        for i in range(0, self.popsize):
            print(pop[i][:])
            Robust_accuracy, l2_distance = main_worker(self.args, self.model, pop[i][:])
            # indi = [2,1,0,1,4,2,3]
            # print(indi)
            # Robust_accuracy, l2_distance = main_worker(self.args, self.model, indi)
            function1.append(Robust_accuracy)
            function2.append(l2_distance)

            print('fitness:', Robust_accuracy, l2_distance)
        return function1, function2

    def crowding_distance(self, args, values, front):
        """
        :param values: 群体[目标函数值1，目标函数值2,...]
        :param front: 群体解的等级，类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]]
        :return: front 对应的 拥挤距离
        """
        distance = np.zeros(2 * args.popsize)  # 拥挤距离初始化为0
        for rank in front:  # 遍历每一层Pareto 解 rank为当前等级
            for i in range(len(values)):  # 遍历每一层函数值（先遍历群体函数值1，再遍历群体函数值2...）
                valuesi = [values[i][A] for A in rank]  # 取出rank等级 对应的  目标函数值i 集合
                rank_valuesi = zip(rank, valuesi)  # 将rank,群体函数值i集合在一起
                sort_rank_valuesi = sorted(rank_valuesi, key=lambda x: (x[1], x[0]))  # 先按函数值大小排序，再按序号大小排序

                sort_ranki = [j[0] for j in sort_rank_valuesi]  # 排序后当前等级rank
                sort_valuesi = [j[1] for j in sort_rank_valuesi]  # 排序后当前等级对应的 群体函数值i
                # print(sort_ranki[0],sort_ranki[-1])
                distance[sort_ranki[0]] = np.inf  # rank 等级 中 的最优解 距离为inf
                distance[sort_ranki[-1]] = np.inf  # rank 等级 中 的最差解 距离为inf

                # 计算rank等级中，除去最优解、最差解外。其余解的拥挤距离
                for j in range(1, len(rank) - 2):
                    distance[sort_ranki[j]] = distance[sort_ranki[j]] + (sort_valuesi[j + 1] - sort_valuesi[j - 1]) / (
                            max(sort_valuesi) - min(sort_valuesi))  # 计算距离
        # 按照格式存放distances
        distanceA = [[] for i in range(len(front))]  #
        for j in range(len(front)):  # 遍历每一层Pareto 解 rank为当前等级
            for i in range(len(front[j])):  # 遍历给rank 等级中每个解的序号
                distanceA[j].append(distance[front[j][i]])
        return distanceA

    def fast_non_dominated_sort(self, values1, values2):
        S = [[] for i in range(0, len(values1))]
        front = [[]]
        n = [0 for i in range(0, len(values1))]
        rank = [0 for i in range(0, len(values1))]
        for p in range(0, len(values1)):  # 取出个体p
            S[p] = []
            n[p] = 0
            for q in range(0, len(values1)):  # 对所有其他个体计算与个体p的非支配关系
                if (values1[p] < values1[q] and values2[p] < values2[q]) or (
                        values1[p] <= values1[q] and values2[p] > values2[q]) or (
                        values1[p] > values1[q] and values2[p] <= values2[q]):
                    # p支配q
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] < values1[p] and values2[q] < values2[p]) or (
                        values1[q] <= values1[p] and values2[q] < values2[p]) or (
                        values1[q] < values1[p] and values2[q] <= values2[p]):
                    # q支配p
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)
        i = 0
        while (front[i] != []):
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            front.append(Q)
        del front[len(front) - 1]
        return front

    def gen_children(self, pop, function1_parent, function2_parent):
        pop_size = len(pop)
        for ind in range(pop_size):
            i, j = self.Tournamentselection(function1_parent, function2_parent)
            for k in range(min(len(pop[i]), len(pop[j]))):
                pc = random.uniform(0, 1)
                if pc < 0.8:
                    temp = pop[j][k]
                    pop[j][k] = pop[i][k]
                    pop[i][k] = temp
            for k in range(len(pop[i])):
                pm = random.uniform(0, 1)
                if pm < 0.7:
                    pop[i][k] = random.randint(self.x_bound_lower, self.x_bound_upper)
        return pop

    def Tournamentselection(self, function1_parent, function2_parent):
        child = []
        for k in range(2):
            pop_size = len(function1_parent)
            i = random.randint(0, pop_size - 1)
            j = random.randint(0, pop_size - 1)
            if function1_parent[i] <= function1_parent[j] and function2_parent[i] <= function2_parent[j]:
                child.append(i)
            else:
                child.append(j)
        child1 = child[0]
        child2 = child[1]
        return child1, child2

    def get_pareto_solution(self, pop, front, function1, function2):
        Number = 0
        Parent_pop = []
        parent_fuction1 = []
        parent_fuction2 = []
        for i in range(len(front)):
            Number = Number + len(front[i])
            if Number > len(pop) / 2:
                break
            for j in range(len(front[i])):
                Parent_pop.append(pop[front[i][j]])
                parent_fuction1.append(function1[front[i][j]])
                parent_fuction2.append(function2[front[i][j]])
        return i, Parent_pop, parent_fuction1, parent_fuction2

    def Selection(self, args, pop, function1, function2):
        front = self.fast_non_dominated_sort(function1, function2)
        number, Parent_pop, parent_fuction1, parent_fuction2 = self.get_pareto_solution(pop, front, function1, function2)
        values = [function1, function2]
        distance = self.crowding_distance(args, values, front)
        distanceA = np.array(distance[number])
        inx = distanceA.argsort()
        j = len(front[number]) - 1
        while len(Parent_pop) < args.popsize:
            Parent_pop.append(pop[front[number][inx[j]]])
            parent_fuction1.append(function1[front[number][inx[j]]])
            parent_fuction2.append(function2[front[number][inx[j]]])
            j = j - 1
        return Parent_pop, parent_fuction1, parent_fuction2

    def parento_plot(self, args, Parent_pop, function1_values2, function2_values2, gen_no):
        plt.clf()
        font_merge = self.fast_non_dominated_sort(function1_values2, function2_values2)
        f1 = [function1_values2[k] for k in font_merge[0]]
        f2 = [function2_values2[k] for k in font_merge[0]]
        firstpareto = [Parent_pop[k] for k in font_merge[0]]
        f2 = [item.cpu().detach().item() for item in f2]
        index = []
        for i in range(len(f2)):
            index.append(f2.index(sorted(f2)[i]))
        for i in range(len(f2) - 1):
            j = index[i]
            k = index[i + 1]
            plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='r')
        plt.xlabel('f2', fontsize=15)
        plt.ylabel('f1', fontsize=15)
        function2_values2 = [item.cpu().detach().item() for item in function2_values2]
        plt.scatter(function2_values2, function1_values2, c='blue')
        # plt.show()
        save_path = '/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-1'
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        plt.savefig('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/figure/VGG-1-1/'+ 'gen_' + str(gen_no) + '_caa.jpg')
        scio.savemat(save_path + '/nsga' + str(gen_no) + '.mat', {'function1_values': f1,
                                                                  'function2_values': f2})
        scio.savemat(save_path + '/first_pareto_' + str(gen_no) + '.mat', {'first_pareto': firstpareto})

    def initialization(self):
        solution = []
        for i in range(int(self.popsize)):
            number = random.randint(3, self.max_length)
            choice = []
            for j in range(int(number)):
                choice.append(random.randint(self.x_bound_lower, self.x_bound_upper))
            solution.append(choice)
        return solution

    def evolve(self):
        Fitness = []
        Parent_pop = self.initialization()
        function1_parent, function2_parent = self.calculate_fitness(Parent_pop)
        print('Initialization succeed.')
        gen_no = 0
        while (gen_no < self.args.max_gen):
            print('gen:', gen_no)
            print(Parent_pop)
            self.parento_plot(self.args, Parent_pop, function1_parent, function2_parent, gen_no)
            # 产生子代个体
            Children_pop = self.gen_children(Parent_pop, function1_parent, function2_parent)
            function1_child, function2_child = self.calculate_fitness(Children_pop)
            # 合并父代和子代种群2N
            Merge_pop = Parent_pop + Children_pop
            function1 = function1_parent + function1_child
            function2 = function2_parent + function2_child
            # 通过非支配排序和拥挤度距离进行选择
            Parent_pop, function1_parent, function2_parent = self.Selection(self.args, Merge_pop, function1, function2)

            print(Parent_pop[0])
            ns = NeighborSearch(Parent_pop[0], self.x_bound_lower, self.x_bound_upper, self.model, self.args)
            INDI = ns.evolve()
            INDI = INDI.tolist()
            Parent_pop[0] = INDI
            gen_no = gen_no + 1
        # plt.plot(Fitness)
        # plt.savefig('fitness.png')
        return Parent_pop



def main_worker(args, model, x):
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

    fitness, l2_distance = evaluate(model, test_loader, attack_names, attacks, args)
    return fitness, l2_distance


def evaluate(model, val_loader, attack_names, attacks, args):
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
    # return time_cost, Dist_norm / (32 * 21)
    # return batch_accuracy * 100, Dist_norm / (32 * 21)

if __name__ == '__main__':
    main()