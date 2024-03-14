import torch
import timm
from PIL import Image
import os
from io import BytesIO
from torchvision import transforms
import argparse
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from uap_utils.utils import Normalize
import tqdm
import numpy as np
import copy
from uap_utils.torch_denoise_tv_chambol import denoise_tv_chambolle_torch
from random import randint, uniform



def randomJPEGcompression(image, qf=75):
    """https://discuss.pytorch.org/t/how-can-i-develop-a-transformation-that-performs-jpeg-compression-with-a-random-qf/43588/5"""
    # outputIoStream = BytesIO()
    # image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    # outputIoStream.seek(0)
    # image = outputIoStream.read()
    image_res = None
    with BytesIO() as output:
        image.save(output, "JPEG", quality=qf, optimice=True)
        output.seek(0)
        image_jpeg = Image.open(output)
        image_res = copy.deepcopy(image_jpeg)
    return image_res


# m = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True, num_classes=0, global_pool='')
# m = timm.create_model('adv_inception_v3', pretrained=True)
# m = timm.create_model("inception_resnet_v2", pretrained=True)
# o = m(torch.randn(2, 3, 224, 224))
# print(f'Unpooled shape: {o.shape}')


def get_args():
    parser = argparse.ArgumentParser("Adversarial valid args")
    parser.add_argument("--data-dir", type=str, default='', help='')
    parser.add_argument("--defense_strategy", type=str, default='tvm', help='')
    parser.add_argument("--batch_size", type=int, default=50, help='')
    args = parser.parse_args()
    return args


def jpeg_defense(images, jpeg_transformation):
    """JPEG compression corruption"""
    images_pl = [torchvision.transforms.ToPILImage()(img) for img in images]
    image_jpeg = torch.stack(list(map(jpeg_transformation, images_pl)), dim=0)
    return image_jpeg


def baussian_blur_defense(images, gaussian_blur_transformation):
    """Gaussian blur corruption"""
    images_pl = [torchvision.transforms.ToPILImage()(img) for img in images]
    image_blur = torch.stack(list(map(gaussian_blur_transformation, images_pl)), dim=0)
    return image_blur


@torch.no_grad()
def evaluate_uaps(uap, model, dataloader):
    phar = tqdm.tqdm(dataloader, disable=True)
    correct = 0
    ori_correct = 0
    fool_num = 0
    n = 0
    for i, (images, labels) in enumerate(phar):
        # adv_images.requires_grad = True
        adv_images = torch.clamp((images.to(device) + uap.to(device)), 0, 1)
        if "jpeg" in args.defense_strategy:
            adv_images = jpeg_defense(adv_images, jpeg_transform)
        elif "gaussian_blur" in args.defense_strategy:
            adv_images = baussian_blur_defense(adv_images, transform)
        elif "tvm" in args.defense_strategy:
            adv_images = denoise_tv_chambolle_torch(adv_images, multichannel=True)
        elif "pixel" in args.defense_strategy:
            adv_images = pixel_deflection_without_map(adv_images)

        images = images.to(device)
        adv_images = adv_images.to(device)
        labels = labels.to(device)
        ori_output = model(images)
        ori_pred = torch.argmax(ori_output, dim=1)

        pred_ori_idx = labels == ori_pred
        ori_correct += pred_ori_idx.sum().item()

        output = model(adv_images)
        pred = torch.argmax(output, dim=1)

        pred_pert_idx = labels == pred

        correct += (pred_pert_idx ^ pred_ori_idx).sum().item()

        fool_num += (ori_pred != pred).sum().item()

        n += images.size(0)
    print("Total:{}, success pred: {}, success attack: {}, fool number: {}".format(n, ori_correct, correct, fool_num))
    return np.round(100 * (ori_correct / n), 2), np.round(100 * (correct / n), 2), np.round(
        100 * (fool_num / n), 2)

if __name__ == "__main__":
    args = get_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cuda'

    if "jpeg" in args.defense_strategy:
        # Transforms
        jpeg_transform = transforms.Compose(
            [
                transforms.Lambda(randomJPEGcompression),
                transforms.ToTensor()
            ]
        )
        model = torchvision.models.vgg16(pretrained=True)
    elif "gaussian_blur" in args.defense_strategy:
        transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(5, 5)),
            transforms.ToTensor()
        ])
        model = torchvision.models.vgg16(pretrained=True)
    elif "adv_inception_v3" in args.defense_strategy:
        model = timm.create_model('adv_inception_v3', pretrained=True)
    elif "env_adv_incep_res_v2" in args.defense_strategy:
        model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True, num_classes=0, global_pool='')
    elif "adv_incep_v2" in args.defense_strategy:
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    elif "tvm" or "pixel" in args.defense_strategy:
        model = torchvision.models.vgg16(pretrained=True)

    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model)
    model = model.to(device)
    model.eval()

    sys_str = "linux"
    if "linux" in sys_str:
        data_path = '/mnt/jfs/wangdonghua/dataset/ImageNet/'
        path = '/mnt/jfs/wangdonghua/dataset/UAPS0504'
    elif "win" in sys_str:
        data_path = 'D:/DataSource/ImageNet/'

    input_size = 224
    traindir = os.path.join(data_path, 'ImageNet10k')
    valdir = os.path.join(data_path, 'val')
    # dataset
    test_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(299), # inception_v3
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])

    test_data = ImageFolder(root=valdir, transform=test_transform)
    # sampler=train_sampler, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=False, num_workers=8)

    # uap = torch.load('F:/Checkpoints/AttackTransfer/CVPRW-EXP/UAPS/our_uap_vgg19_feat2_lr0003_eps10_fr9508.pth')

    uap_files = {
        # "uap2017_vgg16": f"{path}/uap_vgg16_eps10_fr8110.pth",
        # "gd_uap_vgg16_data": f"{path}/gduap_vgg16_imagenet_data_iter_2334_val_imagenet_fr94350.pth",
        # "df_uap_vgg16": f"{path}/df_uap_vgg16_eps10_fr9414.tar",
        # "sgd_uap_vgg16": f"{path}/sgd_uap_vgg16_epsilon10_fr8209.tar",
        # 'our_uap_vgg16': f'{path}/our_uap_vgg16_eps10_fr9589.pth',

        # "uap2017_vgg19": f"{path}/uap_vgg19_eps10_fr8270.pth",
        # "gd_uap_vgg19_data": f"{path}/gduap_vgg19_imagenet_data_iter_1047_val_imagenet_fr94850.pth",
        # "df_uap_vgg19": f"{path}/df_uap_vgg19_eps10_fr9347.tar",
        # "sgd_uap_vgg19": f"{path}/sgd_uap_vgg19_epsilon10_fr7777.tar",
        # 'our_uap_vgg19': f'{path}/our_uap_vgg19_eps10_fr9508.pth',

        # "uap2017_resnet50": f"{path}/uap_resnet50_eps10_fr8380.pth",
        # "gd_uap_resnet50_data": f"{path}/gduap_resnet50_imagenet_data_iter_3772_val_imagenet_fr80500.pth",
        # "df_uap_resnet50": f"{path}/df_uap_resnet50_eps10_fr8639.tar",
        # "sgd_uap_resnet50": f"{path}/sgd_uap_resnet50_epsilon10_fr7340.tar",
        # 'our_uap_resnet50': f'{path}/our_uap_resnet50_eps10_fr9160.pth',

        # "uap2017_resnet101": f"{path}/uap_resnet101_eps10_fr8015.pth",
        # "gd_uap_resnet101_data": f"{path}/gduap_resnet101_imagenet_data_iter_3360_val_imagenet_fr72350.pth",
        # "df_uap_resnet101": f"{path}/df_uap_resnet101_eps10_fr8669.tar",
        # "sgd_uap_resnet101": f"{path}/sgd_uap_resnet101_epsilon10_fr2429.0.tar",
        # 'our_uap_resnet101': f'{path}/our_uap_resnet101_eps10_fr8774.pth',

        # "uap2017_resnet152": f"{path}/uap_resnet152_eps10_fr8025.pth",
        # "gd_uap_resnet152_data": f"{path}/gduap_resnet152_imagenet_data_iter_3176_val_imagenet_fr71300.pth",
        # "df_uap_resnet152": f"{path}/df_uap_resnet152_eps10_fr8394.tar",
        # "sgd_uap_resnet152": f"{path}/sgd_uap_resnet152_epsilon10_fr2380.tar",
        # 'our_uap_resnet152': f'{path}/our_uap_resnet152_eps10_fr8875.pth',

        # "uap2017_resnext50": f"{path}/uap_resnext50_eps10_fr8105.pth",
        # "gd_uap_resnext50_data": f"{path}/gduap_resnext50_imagenet_data_iter_1461_val_imagenet_fr83750.pth",
        # "df_uap_resnext50": f"{path}/df_uap_resnext50_eps10_fr8338.tar",
        # "sgd_uap_resnext50": f"{path}/sgd_uap_resnext50_epsilon10_fr5453.tar",
        # 'our_uap_resnext50': f'{path}/our_uap_resnext50_eps10_fr9255.pth',

        "uap2017_wideresnet50": f"{path}/uap_wideresnet_eps10_fr8030.pth",
        "gd_uap_wideresnet50_data": f"{path}/gduap_wideresnet_imagenet_data_iter_1825_val_imagenet_fr69050.pth",
        "df_uap_wideresnet50": f"{path}/df_uap_wideresnet50_eps10_fr8712.tar",
        "sgd_uap_wideresnet50": f"{path}/sgd_uap_wideresnet50_epsilon10_fr5404.tar",
        'our_uap_wideresnet50': f'{path}/our_uap_wideresnet_eps10_fr9134.pth',

        "uap2017_efficientnetb0": f"{path}/uap_efficientnetb0_eps10_fr8090.pth",
        "gd_uap_efficientnetb0_data": f"{path}/gduap_efficientnetb0_imagenet_data_iter_1202_val_imagenet_fr62700.pth",
        "df_uap_efficientnetb0": f"{path}/df_uap_efficientnet_eps10_fr8276.tar",
        "sgd_uap_efficientnetb0": f"{path}/sgd_uap_efficientnet_epsilon10_fr8286.tar",
        'our_uap_efficientnetb0': f'{path}/our_uap_efficientnetb0_eps10_fr8892.pth',

        "uap2017_densenet121": f"{path}/uap_densenet121_eps10_fr8090.pth",
        "gd_uap_densenet121_data": f"{path}/gduap_densenet121_imagenet_data_iter_2705_val_imagenet_fr78750.pth",
        "df_uap_densenet121": f"{path}/df_uap_densenet121_eps10_fr8689.tar",
        "sgd_uap_densenet121": f"{path}/sgd_uap_densenet121_epsilon10_fr7305.tar",
        'our_uap_densenet121': f'{path}/our_uap_densenet121_eps10_fr9177.pth',

        "uap2017_densenet161": f"{path}/uap_densenet161_eps10_fr8110.pth",
        "gd_uap_densenet161_data": f"{path}/gduap_densenet161_imagenet_data_iter_1195_val_imagenet_fr92200.pth",
        "df_uap_densenet161": f"{path}/df_uap_densenet161_eps10_fr8333.tar",
        "sgd_uap_densenet161": f"{path}/sgd_uap_densenet161_epsilon10_fr7538.tar",
        'our_uap_densenet161': f'{path}/our_uap_densenet161_eps10_fr9413.pth',

        "uap2017_alexnet": f"{path}/uap_alexnet_eps10_fr9015.pth",
        "gd_uap_alexnet_data": f"{path}/gduap_alexnet_imagenet_data_iter_446_val_imagenet_fr93500.pth",
        "df_uap_alexnet": f"{path}/df_uap_alexnet_eps10_fr9066.tar",
        "sgd_uap_alexnet": f"{path}/sgd_uap_alexnet_epsilon10_fr9302.tar",
        'our_uap_alexnet': f'{path}/our_uap_alexnet_eps10_fr9327.pth',

        "uap2017_googlenet": f"{path}/uap_googlenet_eps10_fr8035.pth",
        "gd_uap_googlenet_data": f"{path}/gduap_googlenet_imagenet_data_iter_2058_val_imagenet_fr87300.pth",
        "df_uap_googlenet": f"{path}/df_uap_googlenet_eps10_fr7797.tar",
        "sgd_uap_googlenet": f"{path}/sgd_uap_googlenet_epsilon10_fr5082.tar",
        'our_uap_googlenet': f'{path}/our_uap_googlenet_eps10_fr8992.pth',

        "uap2017_mnasnet10": f"{path}/uap_mnasnet10_eps10_fr8180.pth",
        "gd_uap_mnasnet10_data": f"{path}/gduap_mnasnet10_imagenet_data_iter_817_val_imagenet_fr95700.pth",
        "df_uap_mnasnet10": f"{path}/df_uap_mnasnet_eps10_fr8879.tar",
        "sgd_uap_mnasnet10": f"{path}/sgd_uap_mnasnet10_epsilon10_fr8343.tar",
        'our_uap_mnasnet10': f'{path}/our_uap_mnasnet10_eps10_fr9719.pth',
    }


    curr_acc = []
    curr_asr = []
    curr_fool_rate = []
    # 在不同防御方法上验证对抗性扰动
    for key, value in uap_files.items():
        if value.endswith('.pth'):
                uap = torch.load(value)
        elif value.endswith('.tar'):
            ckpt = torch.load(value)
            try:
                uap = ckpt['state_dict']['uap']
            except:
                uap = ckpt['uap']
        if len(uap.size()) == 4:
            uap = uap.repeat([args.batch_size, 1, 1, 1])
        else:
            uap = uap.unsqueeze(0).repeat([args.batch_size, 1, 1, 1])
        
        print(key, value)
        acc, asr, fool_rate = evaluate_uaps(uap, model, validation_loader)
        curr_acc.append(acc)
        curr_asr.append(asr)
        curr_fool_rate.append(fool_rate)
    print("model", args.defense_strategy)
    print('acc:', curr_acc)
    print('asr:', curr_asr)
    print('fool_rate:', curr_fool_rate)
    print("\n")
