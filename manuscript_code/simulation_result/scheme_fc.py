import matplotlib.pyplot as plt
import numpy as np

x = np.arange(2, 4.1, 0.2)

y1 = [0.0386851412256687, 0.037054735426014174, 0.03569624542167431, 0.03454688353248893,
      0.033561811276118875, 0.032708153288548925, 0.03196125706626434, 0.0313022733826603,
      0.03071654357182456, 0.03019249629612493, 0.02972087541114835]
y2 = [0.0800705858859054, 0.0800705858859054, 0.0800705858859054, 0.0800705858859054,
      0.0800705858859054, 0.0800705858859054, 0.0800705858859054, 0.0800705858859054,
      0.0800705858859054, 0.0800705858859054, 0.0800705858859054]
y3 = [0.03749142210228251, 0.036311159447777866, 0.035284495648145925, 0.03435544873774323,
      0.03344131864261156, 0.032614918376888666, 0.03188893880996528, 0.031247469383148735,
      0.03067736331732006, 0.03016733955291439, 0.029708375715411295]
y4 = [0.03566344219192551, 0.03432052264948104, 0.03337377725460858, 0.03258957815508654,
      0.03232505273570405, 0.03219853694996232, 0.03157538974001771, 0.030960895301396372,
      0.03039012884188293, 0.029879513765283743, 0.029420017405046482]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Edge-cloud Server Computing Capability (Gigacycles/s)", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(2, 4)
plt.ylim([0.025, 0.085])

plt.plot(x, y1, color='b', marker='v', markersize=8, label="Scheme 1")
plt.plot(x, y2, color='c', marker='^', markersize=8, label="Scheme 2")
plt.plot(x, y3, color='r', marker='o', markersize=8, label="Scheme 3")
plt.plot(x, y4, color='g', marker='s', markersize=8, label="Proposed")

plt.xticks(np.arange(2, 4.1, 0.2), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.025, 0.09, 0.01), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Scheme 1", "Scheme 2", "Scheme 3", "Proposed"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)
# plt.savefig("../figure/scheme_fc.pdf")

plt.show()
