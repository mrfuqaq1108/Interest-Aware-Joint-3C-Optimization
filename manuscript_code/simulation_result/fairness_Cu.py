import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


x = np.arange(0, 1.1, 0.1)

ymin = [0.03512884392198347, 0.03378172734154742, 0.032469287709317854, 0.031157400934240104,
        0.029932624976605385, 0.028748475966583795, 0.02772394588375484, 0.02703464219871718,
        0.026584010763530494, 0.026351021917234436, 0.0262304510958]
ymax = [0.03938212930107079, 0.03938212930107079, 0.03938212930107079, 0.03938212930107079,
        0.03938212930107079, 0.03938212930107079, 0.03938212930107079, 0.03938212930107079,
        0.03938212930107079, 0.03938212930107079, 0.03938212930107079]
yf = [0.03749142210228251, 0.03687961031109271, 0.036326875853365484, 0.03589522428366359,
      0.03566344219192551, 0.035241775831577435, 0.03486289112212461, 0.03459930227480733,
      0.03442265591111736, 0.03432985239073338, 0.03428140949329134]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Local Cache Capability", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(0, 1)
plt.ylim([0.026, 0.042])

plt.gca().xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.1, 0.2)))
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

plt.plot(x, ymax, color='r', marker='o', markersize=8, label="Max Cost")
plt.plot(x, yf, color='g', marker='s', markersize=8, label="Fairness")
plt.plot(x, ymin, color='c', marker='^', markersize=8, label="Min Cost")


# plt.bar(x, ymin, width=0.05, color='red', label='min')
# plt.bar(x, [a-b for (a, b) in zip(yf, ymin)], width=0.05, bottom=ymin, color='green', label='fairness')
# plt.bar(x, [a-b for (a, b) in zip(ymax, yf)], width=0.05, bottom=yf, color='blue', label='max')

plt.xticks(np.arange(0, 1.1, 0.2), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.024, 0.043, 0.003), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Max Cost [16, 17]", "Fairness (Proposed)", "Min Cost"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)
# plt.savefig("../figure/fairness_CU.pdf")

plt.show()
