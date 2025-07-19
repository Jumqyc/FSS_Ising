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

for k in range(1,20):
    data.fit_nu(k)
plt.plot([1,19],[1,1],'--',color='black',alpha=0.5)
plt.xlabel("k")
plt.xticks(range(1,20))
plt.ylabel(r"$\nu$")
plt.show()