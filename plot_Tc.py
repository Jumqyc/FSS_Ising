import matplotlib.pyplot as plt

from pklwrite import load

data = load()

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
