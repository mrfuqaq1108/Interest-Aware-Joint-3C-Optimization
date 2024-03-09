import matplotlib.pyplot as plt
import numpy as np

x = ['20', '30', '40']

y1 = [0.06098601 * 1000, 0.04812398 * 1000, 0.04169591 * 1000]
y2 = [0.09671043 * 1000, 0.09498696 * 1000, 0.0942336 * 1000]
y3 = [0.05655965 * 1000, 0.0466444 * 1000, 0.0412501 * 1000]
y4 = [0.05181967 * 1000, 0.04437869 * 1000, 0.03945445 * 1000]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Total Available Bandwidth (MHz)", font1, labelpad=10)
plt.ylabel('Delay (ms)', font1, labelpad=10)

plt.ylim([0, 100])

x_len = np.arange(len(x))
total_width, n = 0.6, 4
width = total_width / n
xticks = x_len - (total_width - width) / 2

plt.bar(xticks, y1, color='b', width=.8 * width, label='y1', edgecolor='black', linewidth=1)
plt.bar(xticks + width, y2, color='c', width=.8 * width, label='y2', edgecolor='black', linewidth=1)
plt.bar(xticks + width * 2, y3, color='r', width=.8 * width, label='y3', edgecolor='black', linewidth=1)
plt.bar(xticks + width * 3, y4, color='g', width=.8 * width, label='y4', edgecolor='black', linewidth=1)

plt.xticks(x_len, x, family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0, 110, 10), family='Times New Roman', fontsize=15)
plt.gca().yaxis.get_offset_text().set_fontsize(15)

plt.tick_params(axis='both', pad=7)
plt.legend(["Scheme 1", "Scheme 2", "Scheme 3", "Proposed"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='y', linestyle=':', linewidth=1)
# plt.savefig("../figure/bar_t_B.pdf")

plt.show()
