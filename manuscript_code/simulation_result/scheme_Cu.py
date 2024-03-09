import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from brokenaxes import brokenaxes


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


x = np.arange(0, 1.1, 0.1)

y1 = [0.0386851412256687, 0.0386851412256687, 0.0386851412256687, 0.0386851412256687,
      0.0386851412256687, 0.0386851412256687, 0.0386851412256687, 0.0386851412256687,
      0.0386851412256687, 0.0386851412256687, 0.0386851412256687]
y2 = [0.0800705858859054, 0.0800705858859054, 0.0800705858859054, 0.0800705858859054,
      0.0800705858859054, 0.0800705858859054, 0.0800705858859054, 0.0800705858859054,
      0.0800705858859054, 0.0800705858859054, 0.0800705858859054]
y3 = [0.03749142210228251, 0.03749142210228251, 0.03749142210228251, 0.03749142210228251,
      0.03749142210228251, 0.03749142210228251, 0.03749142210228251, 0.03749142210228251,
      0.03749142210228251, 0.03749142210228251, 0.03749142210228251]
y4 = [0.03749142210228251, 0.03687961031109271, 0.036326875853365484, 0.03589522428366359,
      0.03566344219192551, 0.035241775831577435, 0.03486289112212461, 0.03459930227480733,
      0.03442265591111736, 0.03432985239073338, 0.03428140949329134]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Local Cache Capability", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(0, 1)
plt.ylim([0.025, 0.085])
# plt = brokenaxes(xlims=(0, 1), ylims=((0.025, 0.045), (0.055, 0.085)), hspace=0.05)

plt.gca().xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 1.1, 0.2)))
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

plt.plot(x, y1, color='b', marker='v', markersize=8, label="Scheme 1")
plt.plot(x, y2, color='c', marker='^', markersize=8, label="Scheme 2")
plt.plot(x, y3, color='r', marker='o', markersize=8, label="Scheme 3")
plt.plot(x, y4, color='g', marker='s', markersize=8, label="Proposed")

plt.xticks(np.arange(0, 1.1, 0.2), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.025, 0.09, 0.01), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Scheme 1", "Scheme 2", "Scheme 3", "Proposed"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)
# plt.savefig("../figure/scheme_CU.pdf")

plt.show()
