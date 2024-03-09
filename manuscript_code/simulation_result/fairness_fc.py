import matplotlib.pyplot as plt
import numpy as np

x = np.arange(2, 4.1, 0.2)

ymin = [0.029932624976605385, 0.029685209787163894, 0.029347149962900866, 0.029712444418203796,
        0.03141792693968733, 0.03111933642015311, 0.030573413137238766, 0.029981171175627184,
        0.02945473387641689, 0.028983711029755055, 0.028559790467759395]
ymax = [0.03938212930107079, 0.03770176353556693, 0.03630145873098039, 0.03511658543479177,
        0.034100979752344385, 0.03322078816088998, 0.03245062051836739, 0.031771060833788625,
        0.031167007780829724, 0.03062653925976123, 0.030140117590799584]
yf = [0.03566344219192551, 0.03432052264948104, 0.03337377725460858, 0.03258957815508654,
      0.03232505273570405, 0.03219853694996232, 0.03157538974001771, 0.030960895301396372,
      0.03039012884188293, 0.029879513765283743, 0.029420017405046482]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Edge-cloud Server Computing Capability (Gigacycles/s)", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(2, 4)
plt.ylim([0.026, 0.041])

plt.plot(x, ymax, color='r', marker='o', markersize=8, label="Max Cost")
plt.plot(x, yf, color='g', marker='s', markersize=8, label="Fairness")
plt.plot(x, ymin, color='c', marker='^', markersize=8, label="Min Cost")

plt.xticks(np.arange(2, 4.1, 0.2), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.026, 0.042, 0.003), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Max Cost [16, 17]", "Fairness (Proposed)", "Min Cost"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)
# plt.savefig("../figure/fairness_fc.pdf")

plt.show()
