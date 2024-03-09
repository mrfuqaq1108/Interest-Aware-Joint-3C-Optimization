import matplotlib.pyplot as plt
import numpy as np

x = np.arange(20, 41, 2)

y1 = [0.04906222993282339, 0.04623176321852197, 0.043873202955777166, 0.041877646875125546,
      0.04016730764498358, 0.0386851412256687, 0.037388364590869916, 0.0362442613116578,
      0.035227385279785875, 0.034317647465143986, 0.03349897664647142]
y2 = [0.08146108621610167, 0.08105733525427503, 0.08073524401341096, 0.08047259794585344,
      0.08025449415622556, 0.0800705858859054, 0.07991347726218492, 0.07977774791954936,
      0.07965933839671066, 0.07955515041567945, 0.07946277960173731]
y3 = [.04549104132709644, 0.04343712116431682, 0.041686861120979135, 0.04012456019313511,
      0.038722708184278476, 0.03749142210228251, 0.036406639959973036, 0.035457170065011284,
      0.03461012919602884, 0.03383646429128993, 0.03313929446640147]
y4 = [0.04166683083282714, 0.03993435510874312, 0.03845291136528277, 0.03719907771956654,
      0.03615991292310362, 0.03566344219192551, 0.03463646606839702, 0.0337374062503907,
      0.033040848296494875, 0.03232559490193505, 0.03169056406065536]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Total Available Bandwidth (MHz)", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(20, 40)
plt.ylim([0.025, 0.085])

plt.plot(x, y1, color='b', marker='v', markersize=8, label="Scheme 1")
plt.plot(x, y2, color='c', marker='^', markersize=8, label="Scheme 2")
plt.plot(x, y3, color='r', marker='o', markersize=8, label="Scheme 3")
plt.plot(x, y4, color='g', marker='s', markersize=8, label="Proposed")

plt.xticks(np.arange(20, 41, 2), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.025, 0.09, 0.01), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Scheme 1", "Scheme 2", "Scheme 3", "Proposed"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)
# plt.savefig("../figure/scheme_B.pdf")

plt.show()
