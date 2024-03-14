import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
rc = {'font.family': 'serif', 'mathtext.fontset': 'stix'}
plt.rcParams.update(rc)
plt.rcParams['mathtext.default'] = 'regular'

my_font = fm.FontProperties(fname="/mnt/jfs/zhengxiaohu/times/times.ttf")
sns.set(color_codes=True)

# first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-0.1-1/first_pareto_19.mat')
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-0.1-1/nsga19.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='r')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy',  fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1,label=r'$\lambda$ = 0.1, c = 1', marker='x', c='red')
#
# first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-0.5-1/first_pareto_19.mat')
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-0.5-1/nsga19.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='b')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy', fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1, label=r'$\lambda$ = 0.5, c = 1', marker = '+', c='blue')
#
# first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-1/first_pareto_19.mat')
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-1/nsga19.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='magenta')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy', fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1, label=r'$\lambda$ = 1, c = 1', marker='>', c='magenta')
#
#
# first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-5-1/first_pareto_19.mat')
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-5-1/nsga19.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='green')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy', fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1, label=r'$\lambda$ = 5, c = 1', marker='d', c='green')
#
# plt.legend(fontsize=12, loc='upper right')
# # plt.show()
# plt.savefig('hyper1.pdf')


# first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-1/first_pareto_19.mat')
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-1/nsga19.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='r')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy',  fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1,label=r'$\lambda$ = 1, c = 1', marker='x', c='red')
#
# first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-3/first_pareto_19.mat')
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-3/nsga19.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='b')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy', fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1, label=r'$\lambda$ = 1, c = 3', marker = '+', c='blue')
#
# first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-5/first_pareto_19.mat')
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-5/nsga19.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='magenta')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy', fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1, label=r'$\lambda$ = 1, c = 5', marker='>', c='magenta')
#
#
# first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-7/first_pareto_19.mat')
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG-1-7/nsga19.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='green')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy', fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1, label=r'$\lambda$ = 1, c = 7', marker='d', c='green')
#
# plt.legend(fontsize=12, loc='upper right')
# # plt.show()
# plt.savefig('hyper2.pdf')



# #画第一代和第20代的帕累托前沿对比图
nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG/nsga0.mat')

f2 = nsga['function2_values'][0]
f1 = nsga['function1_values'][0]
print(f1)
print(f2)
f2 = [item.item() for item in f2]
index = []
for i in range(len(f2)):
    index.append(f2.index(sorted(f2)[i]))
for i in range(len(f2) - 1):
    j = index[i]
    k = index[i + 1]
    plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='r')
plt.xlabel('L2', fontsize=15)
plt.ylabel('Robust Accuracy',  fontsize=15)
f1 = [item.item() for item in f1]
plt.scatter(f2, f1,label=r'gen=1', marker='D', c='black')

nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/result/VGG/nsga19.mat')

f2 = nsga['function2_values'][0]
f1 = nsga['function1_values'][0]
print(f1)
print(f2)
f2 = [item.item() for item in f2]
index = []
for i in range(len(f2)):
    index.append(f2.index(sorted(f2)[i]))
for i in range(len(f2) - 1):
    j = index[i]
    k = index[i + 1]
    plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='green')
plt.xlabel('L2 + T', fontsize=15)
plt.ylabel('Robust Accuracy', fontsize=15)
f1 = [item.item() for item in f1]
plt.scatter(f2, f1, label=r'gen=20', marker = 'h', c='blue')
plt.legend(fontsize=12, loc='upper right')
# plt.show()
plt.savefig('ec.pdf')


#画第一代和第20代的帕累托前沿对比图
# nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/evocomposite/Initial1/nsga0.mat')
#
# f2 = nsga['function2_values'][0]
# f1 = nsga['function1_values'][0]
# print(f1)
# print(f2)
# f2 = [item.item() for item in f2]
# index = []
# for i in range(len(f2)):
#     index.append(f2.index(sorted(f2)[i]))
# for i in range(len(f2) - 1):
#     j = index[i]
#     k = index[i + 1]
#     plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='r')
# plt.xlabel('L2', fontsize=15)
# plt.ylabel('Robust Accuracy',  fontsize=15)
# f1 = [item.item() for item in f1]
# plt.scatter(f2, f1,label=r'gen=1', marker='D', c='black')
# plt.legend(fontsize=12, loc='upper right')
# plt.show()
# plt.savefig('ec.pdf')

