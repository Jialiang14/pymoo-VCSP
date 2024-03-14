# MES-VCSP绘图
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

rc = {'font.family': 'serif', 'mathtext.fontset': 'stix'}
plt.rcParams.update(rc)
plt.rcParams['mathtext.default'] = 'regular'

my_font = fm.FontProperties(fname="/mnt/jfs/zhengxiaohu/times/times.ttf")
sns.set(color_codes=True)
# name_list = ['VGG', 'RN18','GN', 'DN121', 'MNV2','WRN','PARN']
# num_list = [58.9, 64.5,66.4, 65.6, 62.0,71.1,68.8]
# num_list1 = [54.5, 59.7,61.4, 60.9, 57.9,65.1,63.0]
# num_list = [53.6, 58.2,60.7, 60.5, 55.7,65.5,62.9]
# num_list1 = [47.5, 51.2, 53.3, 54.3, 50.4,57.1,55.0]

name_list = ['GAT', 'EAT','LS', 'FAT', 'RT']
# num_list = [48.2, 31.2, 21.2, 1.8, 8.3]
# num_list1 = [30.5, 24.2, 11.9, 1.5, 3.7]
num_list = [36.2, 10.2, 1.8, 0.4, 2.5]
num_list1 = [24.5, 8.2, 0.7, 0.7, 3.1]
x = list(range(len(num_list)))
# plt.figure(figsize=(6,5))
plt.figure(figsize=(16,16))
plt.subplot(221)
total_width, n = 0.6, 2
width = total_width / n
plt.bar(x, num_list, width=width, label="CSP", fc = "#0000FF")
for a,b in zip(x,num_list):
    plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label="MES-VCSP", tick_label = name_list, fc ="#FF0000")
for a,b in zip(x,num_list1):
 plt.text(a,b,'%.1f'%b,ha='center',va='bottom',fontsize=12)

plt.xlabel("(a)",fontsize=20)
plt.ylabel("Robust accuracy(%)",fontsize=20)
# plt.title("Surrogate model: VGG-16",fontsize=10)
# plt.xticks(fontsize=14, rotation = 45)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0,40)
plt.legend(fontsize=14, loc='upper right')

# name_list = ['VGG', 'RN18','GN', 'DN121', 'MNV2','WRN','PARN']
# num_list = [17.0, 18.9, 16.0, 13.6, 17.1,14.8,15.5]
# num_list1 = [23.3, 25.1, 22.6, 20.1, 22.8,22.2,23.0]
# num_list = [24.3, 26.7, 23.1, 20.2, 25.4,21.4,22.9]
# num_list1 = [33.0, 35.6, 32.6, 28.4, 32.5,31.5,32.4]

name_list = ['GAT', 'EAT','LS', 'FAT', 'RT']
# num_list = [30.3, 63.4, 70.3, 95.2, 87.6]
# num_list1 = [48.3, 72.2, 80.6, 98.3, 94.3]
num_list = [42.7, 85.2, 96.1, 97.2, 96.2]
num_list1 = [62.8, 89.9, 99.2, 98.9, 98.1]
x = list(range(len(num_list)))
plt.subplot(222)
total_width, n = 0.6, 2
width = total_width / n
plt.bar(x, num_list, width=width, label="CSP", fc="#0000FF")
for a, b in zip(x, num_list):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label="MES-VCSP", tick_label=name_list, fc="#FF0000")
for a, b in zip(x, num_list1):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=12)


plt.xlabel("(b)", fontsize=20)
plt.ylabel("Attack succss rate(%)", fontsize=20)
# plt.title("Surrogate model: Resnet18", fontsize=10)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0,120)
plt.legend(fontsize=14, loc='upper right')


# name_list = ['VGG', 'RN18','GN', 'DN121', 'MNV2','WRN','PARN']
# num_list = [38.98, 51.24, 169.52, 200.67, 74.94,224.7,146.7]
# num_list1 = [34.10, 43.33, 126.31, 144.46, 60.20,156.1,107.7]
# num_list = [375.85, 529.74, 2024.41, 2121.37, 810.59,2512,1599]
# num_list1 = [313.94, 426.46, 1544.42, 1615.47, 636.7,1909,1245]

name_list = ['GAT', 'EAT','LS', 'FAT', 'RT']
# num_list = [260.74, 490.09, 283.48, 245.26, 352.04]
# num_list1 = [220.50, 420.82, 246.19, 212.01, 261.87]
num_list = [2313.33, 4120.43, 880.51, 421.65, 1280.16]
num_list1 = [1854.96, 3620.89, 670.92, 323.39, 1160.10]

x = list(range(len(num_list)))
plt.subplot(223)
total_width, n = 0.6, 2
width = total_width / n
plt.bar(x, num_list, width=width, label="CSP", fc="#0000FF")
for a, b in zip(x, num_list):
    plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label="MES-VCSP", tick_label=name_list, fc="#FF0000")
for a, b in zip(x, num_list1):
    plt.text(a, b, '%.0f' % b , ha='center', va='bottom', fontsize=12)


plt.xlabel("(c)", fontsize=20)
plt.ylabel("Time cost(s)", fontsize=20)
# plt.title("Surrogate model: Resnet18", fontsize=10)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0,4550)
plt.legend(fontsize=14, loc='upper right')

# name_list = ['VGG', 'RN18','GN', 'DN121', 'MNV2','WRN','PARN']
# num_list = [12.37, 13.01, 12.72, 12.47, 12.46,12.7,12.5]
# num_list1 = [11.37, 12.64, 11.27, 11.30, 11.89,12.1,11.7]

# num_list = [12.80, 13.96, 13.27, 12.92, 13.17,13.4,12.3]
# num_list1 = [11.72, 12.69, 12.11, 11.97, 12.16,12.4,12.2]

name_list = ['GAT', 'EAT','LS', 'FAT', 'RT']
# num_list = [98.50, 104.54, 94.82, 83.46, 89.39]
# num_list1 = [77.23, 92.46, 75.74, 65.66, 76.65]
#
num_list = [86.13, 107.26, 86.43, 76.21, 87.43]
num_list1 = [74.02, 98.02, 89.27, 61.74, 80.28]

x = list(range(len(num_list)))
plt.subplot(224)
total_width, n = 0.6, 2
width = total_width / n
plt.bar(x, num_list, width=width, label="CSP", fc="#0000FF")
for a, b in zip(x, num_list):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
p1 = plt.bar(x, num_list1, width=width, label="MES-VCSP", tick_label=name_list, fc="#FF0000")
for a, b in zip(x, num_list1):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=12)


plt.xlabel("(d)", fontsize=20)
plt.ylabel(r'$L_{2}$', fontproperties=my_font, fontsize=20)
# plt.title("Surrogate model: Resnet18", fontsize=10)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0,120)
# plt.ylim(0,16)
plt.legend(fontsize=14, loc='upper right')

# plt.legend(fontsize=10, loc='upper left')
plt.savefig('defenseimagenetsche.pdf')
# plt.savefig('defensecifar10sche.pdf')
# plt.savefig(fname="AllConv_T",format="svg")
plt.show()


