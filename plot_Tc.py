from fss import fss
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

data = fss()
for file in os.listdir("data"):
    if file.endswith(".pkl"):
        with open(os.path.join("data", file), "rb") as f:
            model = pkl.load(f)
        size = model.get_spin().shape[0]
        temperature = model.get_temperature()
        data.add_raw_data(model.get_e(), np.abs(model.get_m()), temperature, size)

print('Fit using susceptibility')
data.fit_Tc('(<m**2>-<m>**2)/T/L**2')

print('Fit using Heat Capacity')
data.fit_Tc('(<e**2>-<e>**2)/T**2/L**2')

print('Fit using 1-order Binder Cumulant')
data.fit_Tc(data.binderd[1])

print('Fit using 2-order Binder Cumulant')
data.fit_Tc(data.binderd[2])

print('Fit using <m>')
data.fit_Tc(data.logd[1])

print('Fit using <m**2>')
data.fit_Tc(data.binderd[2])