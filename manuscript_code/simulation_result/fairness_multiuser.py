import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5, 51, 5)

ymin = [0.029932624976605385, 0.03363479885741076, 0.03733697273821614, 0.04103914661902151,
        0.04474132049982689, 0.048443494380632265, 0.05214566826143765, 0.05584784214224302,
        0.0595500160230484, 0.06325218990385378]
ymax = [0.03938212930107079, 0.06028023518159917, 0.08117834106212758, 0.09641260643512063,
        0.09797382773706491, 0.10334886115165273, 0.10909496755132203, 0.11370062312140927,
        0.11817115939262543, 0.12264169566384162]
yf = [0.03566344219192551, 0.046086460546902686, 0.05539223281685748, 0.06220093980748959,
      0.06924740959358404, 0.07616864286997299, 0.08334574281503307, 0.09067443691620261,
      0.09597716891177796, 0.09868761342853163]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Number of Users", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(5, 50)
plt.ylim([0.020, 0.125])

plt.plot(x, ymax, color='r', marker='o', markersize=8, label="Max Cost")
plt.plot(x, yf, color='g', marker='s', markersize=8, label="Fairness")
plt.plot(x, ymin, color='c', marker='^', markersize=8, label="Min Cost")

plt.xticks(np.arange(5, 51, 5), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.020, 0.126, 0.015), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Max Cost [16, 17]", "Fairness (Proposed)", "Min Cost"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)
# plt.savefig("../figure/fairness_multiuser.pdf")

plt.show()
