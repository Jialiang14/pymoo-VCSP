from __future__ import print_function
import os
import argparse
import shutil
import builtins
import csv
import random
import utils
import torchvision.datasets as dset
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import transforms
from optimizer_adv.SE_RNR_MP.composite.composite_adv.attacks import *
from optimizer_adv.SE_RNR_MP.composite.composite_adv.utilities import make_dataloader, EvalModel
import numpy as np
from model_search import Network
import warnings
warnings.filterwarnings('ignore')


def list_type(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("List must be (x,x,....,x) ")


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='wideresnet',
                    help='architecture of model')
parser.add_argument('--model-dir', default='./model-cifar',
                    help='directory of model for saving checkpoint')
parser.add_argument('--mode', default='adv_train_madry', type=str,
                    help='specify training mode (natural or adv_train)')
parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--order', default='random', type=str, help='specify the order')
parser.add_argument("--enable", type=list_type, default=(0, 1, 2, 3, 4, 5), help="list of enabled attacks")
parser.add_argument("--log_filename", default='logfile.csv', help="filename of output log")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:9527', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--num-batches', type=int, required=False,
                    help='number of batches (default entire dataset)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset name')
parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data',
                    help='path to datasets directory')

start_num = 1
iter_num = 1
inner_iter_num = 10

no_improve = 0

classes_map = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main(genotype):
    # settings
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    CIFAR_CLASSES = 10
    torch.cuda.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()

    if args.seed is not None:
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    acc = main_worker(args.gpu, ngpus_per_node, args, model)
    return acc


def main_worker(gpu, ngpus_per_node, args, model):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Send to GPU
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.dataset == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=2)
    acc = train(model, optimizer, criterion, train_queue, valid_queue, args, ngpus_per_node)
    return acc


def train_ep(args, epoch, model, train_loader, composite_attack, optimizer, criterion):
    # global epoch
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.gpu is not None:
            data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        elif torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        # clean training
        if args.mode == 'natural':
            # zero gradient
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            raise ValueError()

        # adv training normal
        elif args.mode == 'adv_train_madry':
            model.eval()
            # generate adversarial example
            if args.gpu is not None:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda(args.gpu, non_blocking=True).detach()
            else:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()

            data_adv = composite_attack(data_adv, target)
            data_adv = Variable(torch.clamp(data_adv, 0.0, 1.0), requires_grad=False)

            model.train()

            # zero gradient
            optimizer.zero_grad()
            logits = model(data_adv)
            loss = criterion(logits, target)

        # adv training by trades
        elif args.mode == 'adv_train_trades':
            # TRADE Loss would require more memory.

            model.eval()
            batch_size = len(data)
            # generate adversarial example
            if args.gpu is not None:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda(args.gpu, non_blocking=True).detach()
            else:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()

            data_adv = composite_attack(data_adv, target)
            data_adv = Variable(torch.clamp(data_adv, 0.0, 1.0), requires_grad=False)

            model.train()
            # zero gradient
            optimizer.zero_grad()

            # calculate robust loss
            logits = model(data)
            loss_natural = F.cross_entropy(logits, target)
            loss_robust = (1.0 / batch_size) * F.kl_div(F.log_softmax(model(data_adv), dim=1),
                                                        F.softmax(model(data), dim=1))
            loss = loss_natural + args.beta * loss_robust

        else:
            print("Not Specify Training Mode.")
            raise ValueError()

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                    100. * batch_idx / len(train_loader), loss.item()))


def train(model, optimizer, criterion, train_loader, test_loader, args, ngpus_per_node):
    # global best_acc1, epoch, no_improve

    composite_attack = CompositeAttack(model, args.enable, mode='train', local_rank=args.rank,
                                       start_num=start_num, iter_num=iter_num,
                                       inner_iter_num=inner_iter_num, multiple_rand_start=True, order_schedule=args.order)
    best_acc1 = 0
    no_improve = 0
    for e in range(0, args.epochs):
        epoch = e
        # adjust learning rate for SGD
        adjust_learning_rate(epoch, optimizer, args)

        # adversarial training
        train_ep(args, epoch, model, train_loader, composite_attack, optimizer, criterion)

        # evaluation on natural examples
        test_loss, test_acc1 = eval_test(model, test_loader, args)
        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        if is_best:
            no_improve = no_improve - (no_improve % 10)
        else:
            no_improve = no_improve + 1
        print("No improve: {}".format(no_improve))
        print("Best Test Accuracy: {}%".format(best_acc1))
        # evaluation on MP
    acc = evaluate_MP(model, test_loader, args)
    print(acc)
    return acc

def evaluate_MP(model, val_loader, args):
    attacks = []
    attack_names: List[str] = [
    "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=10)" ,
    "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=10)" ,
    "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=10)",
    "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=10)",
    "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=10)",
    "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=20)"]
    for attack_name in attack_names:
        # print(attack_name)
        tmp = eval(attack_name)
        attacks.append(tmp)
    model.eval()

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_ori_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_time_used: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}

    for batch_index, (inputs, labels) in enumerate(val_loader):
        # print(f'BATCH {batch_index:05d}'

        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
            labels = labels.cuda()

        for attack_name, attack in zip(attack_names, attacks):
            batch_tic = time.perf_counter()
            adv_inputs = attack(inputs, labels)

            # ae = adv_inputs[0,:,:,:]
            # images = ae.squeeze(0)
            # print(images.shape)
            # plt.axis('off')
            # plt.imshow(images.permute(1, 2, 0).cpu().detach().numpy())
            # plt.savefig('/mnt/jfs/sunjialiang/AAAD/noise_visualization/CAA/5.png', bbox_inches='tight', pad_inches=0.02)

            with torch.no_grad():
                ori_logits = model(inputs)
                adv_logits = model(adv_inputs)
            batch_ori_correct = (ori_logits.argmax(1) == labels).detach()
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            batch_attack_success_rate = 1.0 - batch_correct[batch_ori_correct].float().mean().item()
            batch_toc = time.perf_counter()
            time_used = torch.tensor(batch_toc - batch_tic)
            # print(f'ATTACK {attack_name}',
            #       f'accuracy = {batch_accuracy * 100:.1f}',
            #       f'attack_success_rate = {batch_attack_success_rate * 100:.1f}',
            #       f'time_usage = {time_used:0.2f} s',
            #       sep='\t')
            batches_ori_correct[attack_name].append(batch_ori_correct)
            batches_correct[attack_name].append(batch_correct)
            batches_time_used[attack_name].append(time_used)
    # print('OVERALL')
    accuracies = []
    attack_success_rates = []
    total_time_used = []
    ori_correct: Dict[str, torch.Tensor] = {}
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        ori_correct[attack_name] = torch.cat(batches_ori_correct[attack_name])
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        attack_success_rate = 1.0 - attacks_correct[attack_name][ori_correct[attack_name]].float().mean().item()
        time_used = sum(batches_time_used[attack_name]).item()
        # print(f'ATTACK {attack_name}',
        #       f'accuracy = {accuracy * 100:.1f}',
        #       f'attack_success_rate = {attack_success_rate * 100:.1f}',
        #       f'time_usage = {time_used:0.2f} s',
        #       sep='\t')
        accuracies.append(accuracy)
        attack_success_rates.append(attack_success_rate)
        total_time_used.append(time_used)
    return accuracies

def eval_test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            elif torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(epoch, optimizer, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch < 99:
        lr = args.lr * 0.1
    elif epoch >= 75:
        lr = args.lr * 0.1
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
