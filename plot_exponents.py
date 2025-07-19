import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
from fss import fss
    
data = fss()
for file in os.listdir("data"):
    if file.endswith(".pkl"):
        with open(os.path.join("data", file), "rb") as f:
            model = pkl.load(f)
        size = model.get_spin().shape[0]
        temperature = model.get_temperature()
        data.add_raw_data(model.get_e(), np.abs(model.get_m()), temperature, size)

print('bound for y_t')
data.fit_a(data.logd[2])
data.fit_a(data.logd[1])
data.fit_a(data.binderd[2])
data.fit_a(data.binderd[1])

print('bound for 2y_h-2')
data.fit_a('(<m**2>-<m>**2)/T/L**2')

print('bound for 2y_t-2')
data.fit_a('(<e**2>-<e>**2)/T**2/L**2')

print('bound for y_h+y_t -2')
data.fit_a('(<e*m>-<e>*<m>)/T**2')

# x-axis :y_t
# y-axis :y_h
plt.figure()
plt.xlim(0.9, 1.1)
plt.ylim(1.7, 2)

plt.plot([1.009-0.024,1.009-0.024], [1.7, 2], 'k--',color = 'black',label = r'Estimation using $\log M^k$')
plt.plot([0.990+0.019,0.990+0.019], [1.7, 2], 'k--',color = 'black')

# 1.745-0.015<2y_h-2< 1.745+0.015
plt.plot([0.9,1.1],np.array([1.745-0.015, 1.745-0.015])/2+1, 'k--',color = 'blue', label = r'Estimation using $\chi$')
plt.plot([0.9,1.1],np.array([1.745+0.015, 1.745+0.015])/2+1, 'k--',color = 'blue')

# 2.865-0.028<y_h+y_t-2< 2.865+0.028
y_t_val = np.linspace(0.9, 1.1, 100)
y_h_val_down = 2.865-0.028-y_t_val
y_h_val_up = 2.865+0.028-y_t_val
print(y_h_val_down, y_h_val_up)
plt.plot(y_t_val, y_h_val_down, 'k--',color = 'green', label = r'Estimation using $\langle em\rangle-\langle e\rangle\langle m\rangle $')
plt.plot(y_t_val, y_h_val_up, 'k--',color = 'green')

plt.plot(1,15/8, 'ro', label='Expected')
plt.legend()
plt.xlabel(r'$y_t$')
plt.ylabel(r'$y_h$')
plt.title('Bounds for $y_t$ and $y_h$')
plt.show()