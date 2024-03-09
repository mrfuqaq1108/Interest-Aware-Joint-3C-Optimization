import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


x = np.arange(0, 1.1, 0.1)

yreal = [0.03749142210228251, 0.03687961031109271, 0.036326875853365484, 0.03589522428366359,
         0.03566344219192551, 0.035241775831577435, 0.03486289112212461, 0.03459930227480733,
         0.03442265591111736, 0.03432985239073338, 0.03428140949329134]
yavg = [0.03749142210228251, 0.03715882670607857, 0.036642399635915146, 0.03626192509087999,
        0.035924397157067085, 0.035651206393816574, 0.035569389528736776, 0.035526821010698664,
        0.03505243311444784, 0.0347941727578116, 0.03428140949329134]
yzif = [0.03749142210228251, 0.03687961031109271, 0.03670817445796764, 0.03663437939525196,
        0.036212036144741916, 0.03606205323318508, 0.035654691560065135, 0.0352179158526593,
        0.03505263754266968, 0.03455460924352787, 0.03428140949329134]
yrand = [0.03749142210228251, 0.03689049829583243, 0.036366463522912, 0.03597856736642704,
         0.03570801166833633, 0.035665947284363814, 0.035302152209983784, 0.03513822051029392,
         0.03464501061921171, 0.03455460924352787, 0.03428140949329134]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Local Cache Capability", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(0, 1)
plt.ylim([0.034, 0.038])

plt.gca().xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.1, 0.2)))
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

plt.plot(x, yreal, color='g', marker='s', markersize=8, label="Proposed")
plt.plot(x, yavg, color='c', marker='^', markersize=8, label="Uniform")
plt.plot(x, yzif, color='r', marker='v', markersize=8, label="Zipf")
plt.plot(x, yrand, color='b', marker='o', markersize=8, label="Random")

plt.xticks(np.arange(0, 1.1, 0.2), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.034, 0.0385, 0.0005), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Proposed", "Uniform [20]", "Zipf [16]", "Random"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)

# plt.savefig("../figure/different_prob.pdf")

plt.show()
