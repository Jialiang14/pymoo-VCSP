import argparse
import csv
import os
import random
import time
import builtins
from typing import Dict, List
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from composite_adv.attacks import *
from composite_adv.utilities import make_dataloader, EvalModel
from math import pi
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
parser.add_argument('--arch', type=str, default='madry_adv_resnet50',
                    help='model architecture')
parser.add_argument('--checkpoint', type=str, default='/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/cifar_gat_finetune_trades_madry_loss_cpu.pt',
                    help='checkpoint path')
parser.add_argument('--input-normalized', default=True, action='store_true',
                    help='model is trained for normalized data')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset name')
parser.add_argument('--dataset-path', type=str, default='/mnt/jfs/sunjialiang/data',
                    help='path to datasets directory')
parser.add_argument('--batch-size', type=int, default=64,
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

    main_worker(args.gpu, args)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))

    from composite_adv.utilities import make_model
    # base_model = make_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # Uncomment the following if you want to load their checkpoint to finetuning
    # from composite_adv.utilities import make_madry_model, make_trades_model
    # base_model = make_madry_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # base_model = make_trades_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # base_model = make_pat_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)
    # base_model = make_fast_at_model(args.arch, args.dataset, checkpoint_path=args.checkpoint)

    # model = EvalModel(base_model, normalize_param=dataset_normalizer[args.dataset],
    #                   input_normalized=args.input_normalized)
    # model = EvalModel(base_model,
    #                   normalize_param={'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    #                   input_normalized=True)
    if args.arch == 'madry_adv_resnet50':
        from AAA.model.models.cifar_models.resnet import resnet50

        model = resnet50()
        model = model.cuda()
        model.load_state_dict({k[13:]: v for k, v in
                               torch.load('/mnt/jfs/sunjialiang/CAA/checkpoints/cifar_linf_8.pt')[
                                   'state_dict'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model)

    # attack_names: List[str] = "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=10)"
    attacks = []
    # for attack_name in attack_names:
    attack_names: List[str] = [
        # "NoAttack()"]
    # "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=20)" ,
    # "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='random', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='scheduled', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='random', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='scheduled', inner_iter_num=10)",
    # "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='random', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='scheduled', inner_iter_num=10)",
    # "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='random', inner_iter_num=10)" ,
    "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='scheduled', inner_iter_num=10)" ]
    #                            "CompositeAttack(model, enabled_attack=(1,0,2,0,1,1), order_schedule='fixed', inner_iter_num=1)"]
    #     "CompositeAttack(model, enabled_attack=(0,2,1), order_schedule='fixed', inner_iter_num=3)"]
    # "CompositeAttack(model, enabled_attack=(0,1,2), order_schedule='scheduled', inner_iter_num=5)" ,
    # "CompositeAttack(model, enabled_attack=(1,0,2,0,1, 1), order_schedule='fixed', inner_iter_num=5)"]
    # tmp = eval("CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=10)")
    for attack_name in attack_names:
        print(attack_name)
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

    evaluate(model, test_loader, attack_names, attacks, args)
    # eval_test(model, test_loader,args)

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
        print(f'BATCH {batch_index:05d}')

        if (
                args.num_batches is not None and
                batch_index >= args.num_batches
        ):
            break

        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
        else:
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
            print(images.shape)
            # plt.axis('off')
            # plt.imshow(images.permute(1, 2, 0).cpu().detach().numpy())
            # plt.show()
            # plt.savefig('/mnt/jfs/sunjialiang/AAAD/noise_visualization/CAA/test.png', bbox_inches='tight', pad_inches=0.02)

            with torch.no_grad():
                ori_logits = model(inputs)
                adv_logits = model(adv_inputs)
            batch_ori_correct = (ori_logits.argmax(1) == labels).detach()
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            batch_attack_success_rate = 1.0 - batch_correct[batch_ori_correct].float().mean().item()
            batch_toc = time.perf_counter()
            time_used = torch.tensor(batch_toc - batch_tic)
            print(dist_norm)
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  f'attack_success_rate = {batch_attack_success_rate * 100:.1f}',
                  f'time_usage = {time_used:0.2f} s',
                  f'l2_norm = {dist_norm:.1f} ',
                  sep='\t')
            batches_ori_correct[attack_name].append(batch_ori_correct)
            batches_correct[attack_name].append(batch_correct)
            batches_time_used[attack_name].append(time_used)

    print('OVERALL')
    accuracies = []
    print(Dist_norm)
    print(count)
    Dist_norm = Dist_norm/count
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
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              f'attack_success_rate = {attack_success_rate * 100:.1f}',
              f'time_usage = {time_used:0.2f} s',
              f'l2_norm = {Dist_norm:.1f} ',
              sep='\t')
        accuracies.append(accuracy)
        attack_success_rates.append(attack_success_rate)
        total_time_used.append(time_used)

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