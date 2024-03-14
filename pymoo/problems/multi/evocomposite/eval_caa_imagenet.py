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
from AAA.optimizer_attack.evocomposite.composite_adv.attacks import *
from AAA.optimizer_attack.evocomposite.composite_adv.utilities import make_dataloader, EvalModel
from math import pi
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
import torchvision.models as models
import gc
import logging
import glob
import timm
import torch.nn as nn
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
parser.add_argument('--batch-size', type=int, default=10,
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

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]
        return x

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    # os.mkdir(path)   # 创建最后一级目录，如果上一级目录不存在，无法创建
    os.makedirs(path)  # 创建多级目录，即使上一级目录不存在，也会创建
  print('Experiment dir : {}'.format(path))

def main():
    # settings
    args = parser.parse_args()
    # args.save = '/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/defensed_imagenet/ls_CSP_fixed_results'
    # create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    # log_format = '%(asctime)s %(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

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


def main_worker(gpu, args):
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


    # standard training
    # model = models.resnet18(pretrained=True)
    # model = models.densenet121(pretrained=True)
    # model = models.vgg19(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.mobilenet_v2(pretrained=True)
    # model = models.inception_v3(pretrained=True)
    model = models.shufflenet_v2_x0_5(pretrained=True)

    # defensed model
    # GAT
    # model = EvalModel(base_model, normalize_param=dataset_normalizer[args.dataset],
    #                   input_normalized=args.input_normalized)

    #  robust transfer
    # from torchvision import models as pt_models
    #
    # model = pt_models.wide_resnet50_2().to("cuda")
    # model.load_state_dict(
    #     torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model/model_weights/DARI/Salman2020Do_50_2.pt"))
    # imagenet_mean = (0.485, 0.456, 0.406)
    # imagenet_std = (0.229, 0.224, 0.225)
    # model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)

     # fastat
    # from torchvision import models as pt_models
    # model = pt_models.resnet50().to("cuda")
    # model.load_state_dict(
    #     torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model/model_weights/FBTF/Wong2020Fast_I.pt"))
    # imagenet_mean = (0.485, 0.456, 0.406)
    # imagenet_std = (0.229, 0.224, 0.225)
    # model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)

    # ensemble-at
    # model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
    # print(model.default_cfg)
    # imagenet_mean = (0.485, 0.456, 0.406)
    # imagenet_std = (0.229, 0.224, 0.225)
    # model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)


    #  label smoothing
    # model = timm.create_model('adv_inception_v3', pretrained=True)
    # imagenet_mean = (0.485, 0.456, 0.406)
    # imagenet_std = (0.229, 0.224, 0.225)
    # model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)

    # model = timm.create_model('inception_resnet_v2', pretrained=True)
    # from torchvision import models as pt_models
    # model = pt_models.resnet50().to("cuda")
    # model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/FBTF/Wong2020Fast_I.pt"))
    # imagenet_mean = (0.485, 0.456, 0.406)
    # imagenet_std = (0.229, 0.224, 0.225)
    # model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
    # attack_names: List[str] = "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=10)"
    attacks = []
    # for attack_name in attack_names:
    attack_names: List[str] = [
        # "NoAttack()"]
    # "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='scheduled', inner_iter_num=2)" ]
    # "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(0,2,1, 2), order_schedule='fixed', inner_iter_num=1)" ]
    # "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(0,), order_schedule='scheduled', inner_iter_num=10)" ]
    # "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=20)" ,
    # "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='random', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='scheduled', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='random', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='scheduled', inner_iter_num=10)",
    # "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='random', inner_iter_num=10)" ,
    # "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='scheduled', inner_iter_num=10)",
    "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='fixed', inner_iter_num=2)" ]
    #                            "CompositeAttack(model, enabled_attack=(2,0,1,3,4,1,4), order_schedule='fixed', inner_iter_num=1)"]
    #     "CompositeAttack(model, enabled_attack=(3,4,2,0,1,3,2,1), order_schedule='scheduled', inner_iter_num=1)"]
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

    # evaluate(model, test_loader, attack_names, attacks, args)
    eval_test(model, test_loader,args)

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
            # dist = dist.view(inputs.shape[0], -1)
            # dist_norm = torch.norm(dist, dim=1, keepdim=True)
            # dist_norm = torch.sum(dist_norm)
            # MSE
            # dist_norm = torch.sum((dist) ** 2)
            # Dist_norm = Dist_norm + dist_norm

            # L2
            dist = dist.view(inputs.shape[0], -1)
            # dist_norm = torch.norm(dist, p=2, keepdim=True)
            dist_norm = torch.norm(dist, dim=1, keepdim=True)
            dist_norm = torch.sum(dist_norm)

            Dist_norm = Dist_norm + dist_norm

            # ae = adv_inputs[0,:,:,:]
            # images = ae.squeeze(0)
            # print(images.shape)
            # plt.axis('off')
            # plt.imshow(images.permute(1, 2, 0).cpu().detach().numpy())
            # plt.show()
            # plt.savefig('/mnt/jfs/sunjialiang/AAAD/noise/visualization/ImageNet/CAA/exp/'+str(count)+ 'first.png', bbox_inches='tight', pad_inches=0.02)

            with torch.no_grad():
                ori_logits = model(inputs)
                adv_logits = model(adv_inputs)
            batch_ori_correct = (ori_logits.argmax(1) == labels).detach()
            batch_correct = (adv_logits.argmax(1) == labels).detach()
            batch_accuracy = batch_correct.float().mean().item()
            batch_attack_success_rate = 1.0 - batch_correct[batch_ori_correct].float().mean().item()
            batch_toc = time.perf_counter()
            time_used = torch.tensor(batch_toc - batch_tic)

            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  f'attack_success_rate = {batch_attack_success_rate * 100:.1f}',
                  f'time_usage = {time_used:0.2f} s',
                  f'l2_norm = {dist_norm:.1f} ',
                  sep='\t')

            batches_ori_correct[attack_name].append(batch_ori_correct)
            batches_correct[attack_name].append(batch_correct)
            batches_time_used[attack_name].append(time_used)
        # gc.collect()
        torch.cuda.empty_cache()
        # if count > 10:
        #     break
    print('OVERALL')
    accuracies = []
    Dist_norm = Dist_norm / (count*10)
    print(Dist_norm)
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
              f'l2_norm = {Dist_norm:.2f} ',
              sep='\t')
        logging.info('accuracy: %s attack_success_rate: %s time_usage: %s l2_norm: %s', accuracy, attack_success_rate, time_used, Dist_norm)
        accuracies.append(accuracy)
        attack_success_rates.append(attack_success_rate)
        total_time_used.append(time_used)



if __name__ == '__main__':
    main()