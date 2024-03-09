import numpy as np
import matplotlib.pyplot as plt
from proplot import rc

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

fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(wspace=0, hspace=0.08)

ax1.plot(x, y1, color='b', marker='v', markersize=8, label="Scheme 1")
ax1.plot(x, y3, color='r', marker='o', markersize=8, label="Scheme 3")
ax1.plot(x, y4, color='g', marker='s', markersize=8, label="Proposed")
ax2.plot(x, y2, color='c', marker='^', markersize=8, label="Scheme 2")

ax1.set_xlim(2, 4)
ax1.set_ylim(0.025, 0.046)
ax2.set_ylim(0.075, 0.085)

ax2.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.xaxis.set_visible(False)

# plt.xticks(np.arange(2, 4.1, 0.2), family='Times New Roman', fontsize=15)
# ax1.set_xticks(np.arange(2, 4.1, 0.2), family='Times New Roman')
ax1.tick_params(axis='both', labelsize=15)
ax2.tick_params(axis='both', labelsize=15)


# ax1.grid(ls='--', alpha=0.5, linewidth=1)
# ax2.grid(ls='--', alpha=0.5, linewidth=1)
plt.show()
