import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
name_list = ['VGG-13', 'VGG-19', 'Resnet18',  'Resnet50']
num_list = [77.8, 81.8, 34.4, 30]
num_list1 = [93.4, 94.6, 60.6, 54.4]
num_list2 = [97.2, 98.8, 74, 70.2]
x = list(range(len(num_list)))
# plt.figure(figsize=(6,5))
plt.figure(figsize=(16,6))
plt.subplot(121)
total_width, n = 0.6, 3
width = total_width / n
plt.bar(x, num_list, width=width, label="$\epsilon=0.05$", fc = "#0087cb")
for a,b in zip(x,num_list):
    plt.text(a,b,'%.2f'%b+"%",ha='center',va='bottom',fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label="$\epsilon=0.1$", tick_label = name_list, fc ="#ffa200")
for a,b in zip(x,num_list1):
 plt.text(a,b,'%.2f'%b+"%",ha='center',va='bottom',fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label="$\epsilon=0.15$", fc ="#9966ff")
for a, b in zip(x, num_list2):
     plt.text(a, b, '%.2f' % b + "%", ha='center', va='bottom', fontsize=12)


plt.xlabel("(a). Black-box models",fontsize=20)
plt.ylabel("ASR(%)",fontsize=20)
plt.title("Surrogate model: VGG-16",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=14, loc='upper right')

name_list = ['VGG-13', 'VGG-16', 'VGG-19', 'Resnet50']
num_list = [40.2, 36.8, 38.8, 43.4]
num_list1 = [64.2, 67, 63.4, 69.4]
num_list2 = [76, 76.6, 73.4, 81.8]
x = list(range(len(num_list)))
plt.subplot(122)
total_width, n = 0.6, 3
width = total_width / n
plt.bar(x, num_list, width=width, label="$\epsilon=0.05$", fc="#0087cb")
for a, b in zip(x, num_list):
    plt.text(a, b, '%.2f' % b + "%", ha='center', va='bottom', fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label="$\epsilon=0.1$", tick_label=name_list, fc="#ffa200")
for a, b in zip(x, num_list1):
    plt.text(a, b, '%.2f' % b + "%", ha='center', va='bottom', fontsize=12)

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label="$\epsilon=0.15$", fc="#9966ff")
for a, b in zip(x, num_list2):
    plt.text(a, b, '%.2f' % b + "%", ha='center', va='bottom', fontsize=12)

plt.xlabel("(b). Black-box models", fontsize=20)
plt.ylabel("ASR(%)", fontsize=20)
plt.title("Surrogate model: Resnet18", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=14, loc='upper right')
# plt.legend(fontsize=10, loc='upper left')
plt.savefig('DP.pdf')
# plt.savefig(fname="AllConv_T",format="svg")
plt.show()