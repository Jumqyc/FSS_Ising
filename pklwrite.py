import numpy as np
import os
from Ising import Ising
import pickle as pkl
from fss import fss
from time import time
os.makedirs("data", exist_ok=True)
def load():
    data_processing = fss()
    for file in os.listdir("data"):
        if file.endswith(".pkl"):
            with open(os.path.join("data", file), "rb") as f:
                model = pkl.load(f)
            size = model.get_spin().shape[0]
            temperature = model.get_temperature()
            data_processing.load_raw_data(model.get_e(), np.abs(model.get_m()), temperature, size)
    return data_processing
# load existing data if available
existing_data = load()

t_val = np.arange(2.1,2.3,0.005)
t_val = np.round(t_val, 3)
for size in [256]:
    model_list = []
    for t in t_val:
        if existing_data.is_in(size, t):
            continue
        start_time = time()
        model = Ising(size, t)
        model.run(1000,1000)
        print(f"Size: {size}, Temperature: {t}, Time taken: {time() - start_time:.2f} seconds")
        with open(f"./data/Ising_{size}_{t}.pkl", "wb") as f:
            pkl.dump(model, f)
        del model