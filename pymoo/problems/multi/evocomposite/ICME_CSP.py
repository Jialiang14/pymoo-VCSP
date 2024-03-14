import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

rc = {'font.family': 'serif', 'mathtext.fontset': 'stix'}
plt.rcParams.update(rc)
plt.rcParams['mathtext.default'] = 'regular'

my_font = fm.FontProperties(fname="/mnt/jfs/zhengxiaohu/times/times.ttf")
sns.set(color_codes=True)
name_list = ['fixed', 'scheduled','fixed', 'scheduled']
num_list = [56.3, 51.1, 56.3, 51.1]
num_list1 = [44.4, 39.5, 56.3, 51.1]
x = list(range(len(num_list)))
# plt.figure(figsize=(6,5))
plt.figure(figsize=(16,16))
plt.subplot(221)
total_width, n = 0.6, 4
width = total_width / n
plt.bar(x, num_list, width=width, label="CSP", fc = "#0000FF")
for a,b in zip(x,num_list):
    plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label="MS-VCSP", tick_label = name_list, fc ="#FF0000")
for a,b in zip(x,num_list1):
 plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=12)

plt.xlabel("(a)",fontsize=20)
plt.ylabel("Robust accuracy(%)",fontsize=20)
# plt.title("Surrogate model: VGG-16",fontsize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=14, loc='upper center')

name_list = ['fixed', 'scheduled','fixed', 'scheduled']
num_list = [35.5, 41.3,56.3, 51.1]
num_list1 = [49.1, 54.6,56.3, 51.1]
x = list(range(len(num_list)))
plt.subplot(222)
total_width, n = 0.5, 3
width = total_width / n
plt.bar(x, num_list, width=width, label="CSP", fc="#0000FF")
for a, b in zip(x, num_list):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label="MS-VCSP", tick_label=name_list, fc="#FF0000")
for a, b in zip(x, num_list1):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=12)


plt.xlabel("(b)", fontsize=20)
plt.ylabel("Attack succss rate(%)", fontsize=20)
# plt.title("Surrogate model: Resnet18", fontsize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=14, loc='upper center')


name_list = ['fixed', 'scheduled','fixed', 'scheduled']
num_list = [99, 1153,99, 1153]
num_list1 = [72, 873,99, 1153]

x = list(range(len(num_list)))
plt.subplot(223)
total_width, n = 0.6, 3
width = total_width / n
plt.bar(x, num_list, width=width, label="CSP", fc="#0000FF")
for a, b in zip(x, num_list):
    plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label="MS-VCSP", tick_label=name_list, fc="#FF0000")
for a, b in zip(x, num_list1):
    plt.text(a, b, '%.0f' % b , ha='center', va='bottom', fontsize=12)


plt.xlabel("(c)", fontsize=20)
plt.ylabel("Time cost(s)", fontsize=20)
# plt.title("Surrogate model: Resnet18", fontsize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=14, loc='upper center')

name_list = ['fixed', 'scheduled','fixed', 'scheduled']
num_list = [20.63, 20.40,20.63, 20.40]
num_list1 = [19.49, 18.80,20.63, 20.40]

x = list(range(len(num_list)))
plt.subplot(224)
total_width, n = 0.6, 3
width = total_width / n
plt.bar(x, num_list, width=width, label="CSP", fc="#0000FF")
for a, b in zip(x, num_list):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
p1 = plt.bar(x, num_list1, width=width, label="MS-VCSP", tick_label=name_list, fc="#FF0000")
for a, b in zip(x, num_list1):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=12)


plt.xlabel("(d)", fontsize=20)
plt.ylabel(r'$L_{2}$', fontproperties=my_font, fontsize=20)
# plt.title("Surrogate model: Resnet18", fontsize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=14, loc='upper center')

# plt.legend(fontsize=10, loc='upper left')
plt.savefig('defensecifar10.pdf')
# plt.savefig(fname="AllConv_T",format="svg")
plt.show()


# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# my_font = fm.FontProperties(fname="/mnt/jfs/zhengxiaohu/times/times.ttf")
# x = [0.0001, 0.0005, 0.001, 0.002, 0.003]
# y2 = [100-20.8, 100-23.8, 100-24.4,100-35.6, 100-36.4]
# y1 = [100-31.2, 100-25.4,100-35.0,100-48.2,100-60.6]
# y3 = [100-13.2, 100-10,100-15,100-19,100-18.4]
# # y2 = [426.31, 298.61, 226.56,199.19, 186.53]
# # y1 = [531.65, 327.00,289.72,270.91,258.28]
# # y3 = [864.38, 524.63,391.99,246.00,210.53]
#
# plt.figure(figsize=(12,8))
# # plt.subplot(121)
# plt.plot(x, y1, 'bs-', color='blue', markersize=10, label="WinterValley")
# plt.plot(x, y2, 'ro-', color='red', markersize=10, label="Forest")
# plt.plot(x, y3, 'g^-', color='green', markersize=10, label="Desert")
# # plt.plot(x, y4, 'b*-', color='black', markersize=10, label="cat")
# # plt.plot(x, y5, 'co-', color='cyan', markersize=10, label="deer")
# # plt.plot(x, y6, 'm1-', color='magenta', markersize=10, label="dog")
# # plt.plot(x, y7, 'g+-', color='green', markersize=10, label="frog")
# # plt.plot(x, y8, 'bh-', color='black', markersize=10, label="horse")
# # plt.plot(x, y9, 'bd-', color='blue', markersize=10, label="ship")
# # plt.plot(x, y10, 'r2-', color='red', markersize=10, label="truck")
# # plt.ylabel(r'Heat flux ($W/m^2$)', fontproperties=my_font)
# plt.xlabel(r'$\lambda_{1}$',fontproperties=my_font,fontsize=26)
# plt.ylabel("ASR($\%$)",fontsize=26)
# # plt.title(r"The effect of coefficient $\lambda_{1}$ on MSE",fontsize=26)
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.legend(fontsize=26, bbox_to_anchor=(0.4, 0.3), loc='upper right')
# plt.subplots_adjust(right=0.80)
# # plt.subplot(122)
# # plt.plot(x, y1, 'bs-', color='blue', markersize=8, label="VGG-13")
# # plt.plot(x, y2, 'ro-', color='red', markersize=8, label="VGG-16")
# # plt.plot(x, y3, 'g^-', color='green', markersize=10, label="VGG-19")
# # plt.plot(x, y4, 'b*-', color='black', markersize=10, label="Resnet50")
# # plt.xlabel("(b). The number of iterations",fontsize=20)
# # plt.ylabel("ASR(%)",fontsize=20)
# # plt.title("Surrogate model: Resnet18",fontsize=20)
# # plt.xticks(fontsize=20)
# # plt.yticks(fontsize=20)
# # plt.legend(fontsize=14, loc='best')
# # plt.legend(fontsize=10, loc='upper left')
# plt.tight_layout()
# plt.savefig('ASR.pdf')
# # plt.savefig(fname="AllConv_T",format="svg")
# plt.show()


# import os
# import sys
# import time
# import math
# import torch
# import pickle
# import scipy.io as sio
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
#
# root_path = "/mnt/jfs/zhengxiaohu/Deep_aPCE/ACAR/"
#
# data = root_path + 'data/ACAR_T_p.mat'
# data = sio.loadmat(data)
# T_p = data['T_p'][0:71]
#
# rc = {'font.family': 'serif', 'mathtext.fontset': 'stix'}
# plt.rcParams.update(rc)
# plt.rcParams['mathtext.default'] = 'regular'
#
# x = np.arange(len(T_p))
# my_font = fm.FontProperties(fname="/mnt/jfs/zhengxiaohu/times/times.ttf")
# fig = plt.figure(figsize=[4, 3], dpi=360)
# plt.plot(x, T_p, color='b')
# plt.xticks(np.arange(0, 71, step=10), fontproperties=my_font)
# plt.yticks(fontproperties=my_font)
# plt.xlabel('Time (second)', fontproperties=my_font)
# plt.ylabel(r'Heat flux ($W/m^2$)', fontproperties=my_font)
# plt.grid(axis='y', alpha=0.1)
#
# plt.tight_layout()
# plt.show()
