import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 15, 1)

yfc2 = [0.2, 0.06452102307197594, 0.00018446526390144408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
yfc3 = [0.2, 0.0676985504426808, 0.00010291260735689062, 0.0002988067202406297, 0.00034657139608103293, 0.0003727613109301177, 1.854523914240208e-05, 0, 0, 0, 0, 0, 0, 0]
yfc4 = [0.2, 0.07057998259495352, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ypr0 = [0.2, 0.0625085778977175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ypr9 = [0.2, 0.06588698819361108, 0.00016839768690241758, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Number of Iterations", font1, labelpad=10)
plt.ylabel('\u0394 Cost', font1, labelpad=10)

plt.xlim(1, 14)
plt.ylim(0, 0.25)

plt.plot(x, yfc2, color='g', marker='s', markersize=8, label="$f^c=2$")
plt.plot(x, yfc3, color='c', marker='^', markersize=8, label="$f^c=3$")
plt.plot(x, yfc4, color='r', marker='v', markersize=8, label="$f^c=4$")
plt.plot(x, ypr0, color='b', marker='o', markersize=8, label="$C_u=0$")
plt.plot(x, ypr9, color='m', marker='d', markersize=8, label="$C_u=10$")

plt.xticks(np.arange(0, 15, 2), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0, 0.26, 0.05), family='Times New Roman', fontsize=15)

plt.gca().yaxis.get_offset_text().set_fontsize(15)

plt.tick_params(axis='both', pad=7)
plt.legend(["$f^c=2~Gigacycles/s, C_u=4$", "$f^c=3~Gigacycles/s, C_u=4$", "$f^c=4~Gigacycles/s, C_u=4$", "$f^c=2~Gigacycles/s, C_u=0$", "$f^c=2~Gigacycles/s, C_u=10$"], prop=font_xy)
plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)

# plt.savefig("../figure/convergence.pdf")

plt.show()
