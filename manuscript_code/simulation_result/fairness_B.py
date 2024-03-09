import matplotlib.pyplot as plt
import numpy as np

x = np.arange(20, 41, 2)

ymin = [0.03178371191700807, 0.03127887002417098, 0.030858168446806722, 0.030502190189036972,
        0.03019706596809148, 0.029932624976605385, 0.029650919290106152, 0.029370569231696377,
        0.02912136917977658, 0.02889840071226939, 0.028660397619847205]
ymax = [0.04983118224133499, 0.04698144053035383, 0.04460665577120289, 0.04259722251345978,
        0.04087485114967996, 0.03938212930107079, 0.03807599768353777, 0.036923528609243916,
        0.03589911165431605, 0.034982528063064816, 0.03415760283093869]
yf = [0.04166683083282714, 0.03993435510874312, 0.03845291136528277, 0.03719907771956654,
      0.03615991292310362, 0.03566344219192551, 0.03463646606839702, 0.0337374062503907,
      0.033040848296494875, 0.03232559490193505, 0.03169056406065536]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Total Available Bandwidth (MHz)", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(20, 40)
plt.ylim([0.026, 0.053])

plt.plot(x, ymax, color='r', marker='o', markersize=8, label="Max Cost")
plt.plot(x, yf, color='g', marker='s', markersize=8, label="Fairness")
plt.plot(x, ymin, color='c', marker='^', markersize=8, label="Min Cost")

plt.xticks(np.arange(20, 41, 2), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.026, 0.054, 0.003), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Max Cost [16, 17]", "Fairness (Proposed)", "Min Cost"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)
# plt.savefig("../figure/fairness_B.pdf")

plt.show()
