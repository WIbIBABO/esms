from math import sqrt
from tensorboardX import SummaryWriter
import torch
import numpy as np
import pandas as pd


def b_s(var, od):
    x = torch.tensor([var], dtype=torch.float64)
    y = torch.tensor([var], dtype=torch.float64)
    bs = torch.tensor([var], dtype=torch.float64)

    for i in range(0, od - 1):
        x = x * y
        bs = torch.cat([bs, x], -1)

    return bs


def f_s(inp, od):
    fs = b_s(inp[0], od[0]).reshape(1, -1)

    for i in range(1, len(inp)):
        cfs = b_s(inp[i], od[i]).reshape(1, -1)
        fs = torch.cat((fs, cfs), -1)

    # fs = torch.reshape(fs, [sum(od)])
    return fs


def f_s_a(inp_a, od):
    fsa = f_s(inp_a[0], od)
    l = len(inp_a)

    for i in range(1, l):
        cfs = f_s(inp_a[i], od)
        fsa = torch.cat((fsa, cfs), 0)

    return fsa


def mulp_sum(inp, od):
    k = od[0]
    mul = 1 + inp[:k].sum()

    for i in range(1, len(od)):
        m = k
        k += od[i]
        mul *= (1 + inp[m:k].sum())

    return mul


def w_b_g(od):
    s = sum(od)
    w = torch.normal(0, 0.01, [s], requires_grad=True, dtype=torch.float64)
    b = torch.rand(1, requires_grad=True)

    return w, b


def i_n(inp, comps):
    a = len(inp)
    b = len(inp[0])
    c = comps
    max = np.zeros(b)
    min = np.zeros(b)
    min += 1000
    for i in range(b):
        for j in range(a):
            if abs(inp[j][i]) > max[i]:
                max[i] = inp[j][i]
            if abs(inp)[j][i] < min[i]:
                min[i] = inp[j][i]
    d = max - min

    for i in range(a):
        for j in range(b):
            inp[i][j] = 2 * (c * ((inp[i][j] - min[j]) / d[j])) - 1

    return inp, max, min, d


def unit_vector(vec):
    s = sqrt(sum((vec * vec)))

    vec /= s
    return vec


torch.set_printoptions(precision=7)

db = np.loadtxt('./TP_pre.csv', dtype=np.float32, encoding='utf-8')
np.random.shuffle(db)

for i in range(8):
    print(db[:, i].sum()/1086)
inputs = db[:, :-1]
inputs, maxm, minm, dm = i_n(inputs, 1)
labels = torch.from_numpy(db[:, -1])

od = [12, 12, 12, 12, 12, 12, 12]
lr = 0.005
amse = 0
max_epoch = 10000
train = 869
test = len(db) - train
o = f_s_a(inputs, od)
w, b = w_b_g(od)
l = len(inputs)
ws = 0
bs = 0
tamse = 0
sreg = 0
MAE = 0
# MAPE = 0

altr = labels[:train].sum()/train
alte = labels[train:].sum()/(len(db)-train)
stottr = ((labels[:train]-altr)**2).sum()
stotte = ((labels[train:]-alte)**2).sum()
# trainloss = SummaryWriter("trainloss")

for epoch in range(max_epoch):
    for i in range(train):
        fout = mulp_sum(w * o[i], od)
        loss = (b * fout - labels[i]) ** 2
        # Writer.add_scalar("test", loss.data, epoch)
        loss.backward()
        amse += loss
        MAE += abs(b * fout - labels[i])
        # MAPE += abs(b * mul - labels[i]) / labels[i]
        ws += w.grad.data
        bs += b.grad.data
        torch.zero_(w.grad.data)
        torch.zero_(b.grad.data)
    ws /= train
    bs /= train
    w.data -= lr * unit_vector(ws)
    b.data -= lr * bs
    ws = 0
    bs = 0
    r2tr = 1 - (amse / stottr)
    amse = sqrt(amse / train)
    MAE /= train
    # MAPE = MAPE / train * 100
    # Writer.add_scalar("RMSE_train_869", amse, epoch)
    # Writer.add_scalar("R2_train_869", r2tr, epoch)
    print('Epoch:{}, train: RMSE={}, R2={}, MAE={}'.format(epoch, amse, r2tr, MAE))
    amse = 0
    MAE = 0

    if (epoch + 1) % 20 == 0:
        for i in range(train, len(db)):
            fsum = mulp_sum(w * o[i], od)
            loss = (b * fsum - labels[i]) ** 2
            amse += loss
            MAE += abs(b * fsum - labels[i])
            # MAPE += abs(b * mul - labels[i]) / labels[i]
        r2te = 1 - (amse / stotte)
        amse = sqrt(amse / test)
        MAE /= test
        # MAPE = MAPE / test * 100
        # Writer.add_scalar("RMSE_train_869", amse, epoch)
        # Writer.add_scalar("R2_train_869", r2tr, epoch)
        print('Epoch:{}, test: RMSE={}, R2={}, MAE={}'.format(epoch, amse, r2te, MAE))
        amse = 0
        MAE = 0

# datab = b.data.numpy()
# dataw = torch.reshape(w, [-1, 12]).data.numpy()
# frameb = pd.DataFrame(datab)
# framew = pd.DataFrame(dataw)
# frameb.to_csv("./results./LESE_para./12order_b6.csv", index=False, header=False, sep=' ')
# framew.to_csv("./results./LESE_para./12order_w6.csv", index=False, header=False, sep=' ')
# Writer.close()
