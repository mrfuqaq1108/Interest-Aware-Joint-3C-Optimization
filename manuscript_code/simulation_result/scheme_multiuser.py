import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5, 51, 5)

y1 = [0.0386851412256687, 0.059441750822305614, 0.08020326752747539, 0.10096603524049383,
      0.12172930729049103, 0.14249283260787254, 0.16325650305477657, 0.18402026438550542,
      0.20478408639264345, 0.22554795092136065]
y2 = [0.0800705858859054, 0.0831120773709252, 0.08728566777447117, 0.09262889965199267,
      0.09896595870088895, 0.10603553278957563, 0.11360845239983386, 0.12152159455388081,
      0.1296674587119505, 0.1379762819683836]
y3 = [0.03749142210228251, 0.05286570815671494, 0.06581045659633981, 0.0775553491901814,
      0.08846636781395464, 0.09908814481652622, 0.10967293215602344, 0.11991468336955259,
      0.12894032700632313, 0.13773578185294783]
y4 = [0.03566344219192551, 0.046086460546902686, 0.05539223281685748, 0.06220093980748959,
      0.06924740959358404, 0.07616864286997299, 0.08334574281503307, 0.09067443691620261,
      0.09597716891177796, 0.09868761342853163]

fig = plt.figure()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Number of Users", font1, labelpad=10)
plt.ylabel('Cost', font1, labelpad=10)

plt.xlim(5, 50)
plt.ylim([0.030, 0.230])

plt.plot(x, y1, color='b', marker='v', markersize=8, label="Scheme 1")
plt.plot(x, y2, color='c', marker='^', markersize=8, label="Scheme 2")
plt.plot(x, y3, color='r', marker='o', markersize=8, label="Scheme 3")
plt.plot(x, y4, color='g', marker='s', markersize=8, label="Proposed")

plt.xticks(np.arange(5, 51, 5), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(0.030, 0.231, 0.05), family='Times New Roman', fontsize=15)

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)
plt.legend(["Scheme 1", "Scheme 2", "Scheme 3", "Proposed"], prop=font_xy)

plt.tight_layout()
plt.grid(axis='both', linestyle=':', linewidth=1)
# plt.savefig("../figure/scheme_multiuser.pdf")

plt.show()
