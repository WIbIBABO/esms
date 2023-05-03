import numpy
import pandas as pd
import torch

db = numpy.loadtxt('./waterdata.csv', dtype=numpy.float32, encoding='utf-8')

Inputs = torch.from_numpy(db[:-1, :])

label = torch.from_numpy(db[1:, -2]).reshape(1085, -1)

data = torch.cat([Inputs, label], -1).numpy()
print(data[1])
frame = pd.DataFrame(data)

frame.to_csv("./TP_pre.csv", index=False, header=False, sep=' ')