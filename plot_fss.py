import matplotlib.pyplot as plt

from pklwrite import load

data = load()

yt = 1
yh = 15/8
data.T_c = 2.26918531421

plt.subplot(2,2,1)
data.a_best = 2-yh
data.b_best = 1/yt
data.plot("<|m|>/L**2", True)
plt.xlim(-10,10)
plt.xlabel("$(T-T_c)L^{y_t}/T_c$")
plt.ylabel("$mL^{y_h-d}$")
plt.xticks([])

plt.subplot(2,2,2)
data.a_best = 2-yh+yt
data.b_best = 1/yt
data.plot("(<e*|m|>-<e>*<|m|>)/T**2/L**4", True)
plt.xticks([])

plt.xlim(-10,10)
plt.xlabel("$(T-T_c)L^{y_t}/T_c$")
plt.ylabel("$m'L^{y_h+y_t-d}$")

plt.subplot(2,2,3)
data.a_best = 2-2*yh
data.b_best = 1/yt
data.plot("(<m**2>-<|m|>**2)/T/L**2", True)
plt.xlim(-10,10)
plt.xlabel("$(T-T_c)L^{y_t}/T_c$")
plt.ylabel("$\chi L^{2y_h-d}$")

plt.subplot(2,2,4)
data.a_best = 2-2*yt
data.b_best = 1/yt
data.plot("(<e**2>-<e>**2)/T**2/L**2", True)
plt.xlim(-10,10)
plt.xlabel("$(T-T_c)L^{y_t}/T_c$")
plt.ylabel("$C L^{2y_t-d}$")

plt.legend()
plt.show()