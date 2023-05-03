import random
import time
import sys
from math import sqrt
import numpy as np
import pandas as pd
import torch
import numpy
import xlrd as xlrd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from tensorboardX import SummaryWriter
from collections import Counter


def b_s(od, var):
    x = torch.tensor([var], dtype=torch.float32)
    y = torch.tensor([var], dtype=torch.float32)
    bs = torch.tensor([1], dtype=torch.float32)

    for i in range(0, od):
        bs = torch.cat([bs, x], -1)
        x = x * y

    return bs


def f_s(od, sqc):
    if len(sqc) > 1:
        fs = b_s(od, sqc[0])
        fs = torch.reshape(fs, [len(fs), 1])

        for i in range(1, len(sqc)):
            cs = b_s(od, sqc[i])
            fs = fs * cs
            fs = torch.reshape(fs, [-1, 1])

        fs = torch.reshape(fs, [-1, ])
    else:
        fs = b_s(od, sqc[0])
    return fs


def f_s_a(inp_a, od, lots):
    fs = f_s(od, inp_a[0])

    fsa = torch.zeros([lots, len(fs)])

    for i in range(lots):
        cfs = f_s(od, inp_a[i])
        fsa[i] = cfs

    return fsa


def w_g(fs):
    w = torch.normal(0, 0.01, [len(fs)], requires_grad=True, dtype=torch.float32)
    # w.retain_grad()

    return w


def f_o(fs, w):
    fo = (fs * w).sum()

    return fo


def i_n(inp, comps):
    a = len(inp)
    b = len(inp[0])
    c = comps
    max = numpy.zeros(b)
    min = numpy.zeros(b)
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
            inp[i][j] = 2*(c * ((inp[i][j] - min[j]) / d[j])) - 1

    return inp


def g_d_f_d(db_p: str, encoding: str):
    db = numpy.loadtxt(db_p, delimiter=',', dtype=numpy.float32, encoding=encoding)
    a = len(db)
    inps = numpy.delete(db, -1, axis=1)
    labs = numpy.zeros(a)

    for i in range(a):
        labs[i] = db[i][-1]

    return inps, labs


def mse(fi_o, label):
    l = 0.5 * (fi_o - label) ** 2

    return l


def stdError_func(y_test, y):
    return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
    return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
    y_mean = np.array(y)
    y_mean[:] = y.mean()
    return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)


def unit_vector(vec):
    s = sqrt(sum((vec * vec)))

    vec /= s
    return vec


torch.set_printoptions(precision=7)

Lr = torch.tensor([0.000007])
Train = 870
max_epoch = torch.tensor([10000])

db = numpy.loadtxt('./waterdata.csv', dtype=numpy.float32, encoding='utf-8')
np.random.shuffle(db)

Inputs = db[:, :-1]
Inputs = i_n(Inputs, 1)
Labels = db[:, -1]
Max_epoch = 10000

Fisa0 = f_s_a(Inputs[:, 0].reshape(-1, 1), 12, len(db))
Fisa1 = f_s_a(Inputs[:, 1].reshape(-1, 1), 12, len(db))
Fisa2 = f_s_a(Inputs[:, 2].reshape(-1, 1), 12, len(db))
Fisa3 = f_s_a(Inputs[:, 3].reshape(-1, 1), 12, len(db))
Fisa4 = f_s_a(Inputs[:, 4].reshape(-1, 1), 12, len(db))
Fisa5 = f_s_a(Inputs[:, 5].reshape(-1, 1), 12, len(db))
Fisa6 = f_s_a(Inputs[:, 6].reshape(-1, 1), 12, len(db))

Fisa0 = torch.cat([Fisa0, Fisa1], -1)
Fisa0 = torch.cat([Fisa0, Fisa2], -1)
Fisa0 = torch.cat([Fisa0, Fisa3], -1)
Fisa0 = torch.cat([Fisa0, Fisa4], -1)
Fisa0 = torch.cat([Fisa0, Fisa5], -1)
Fisa = torch.cat([Fisa0, Fisa6], -1)

lin_reg = linear_model.LinearRegression()


len_er = len(Fisa[0])
# print(Fisa.shape)

trfisa = np.array(Fisa[:Train])
trlabels = Labels[:Train]
tefisa = np.array(Fisa[Train:])
telabels = Labels[Train:]

lin_reg_e = linear_model.LinearRegression()
lin_reg_e.fit(trfisa, trlabels)

pre = lin_reg_e.predict(trfisa)
print(type(pre), type(trlabels))
rmse_er_tr = stdError_func(pre, trlabels)
R2_er_tr = R2_1_func(pre, trlabels)
print('ergo_tr:参数：{}, rmse_er_tr={}, R2_er_tr={}'.format(len_er, rmse_er_tr, R2_er_tr))

pre = lin_reg_e.predict(tefisa)

rmse_er_te = stdError_func(pre, telabels)
R2_er_te = R2_1_func(pre, telabels)
print('ergo_te:参数：{}, rmse_er_te={}, R2_er_te={}'.format(len_er, rmse_er_te, R2_er_te))

altr = Labels[:Train].sum()/Train
alte = Labels[Train:].sum()/(len(db)-Train)
stottr = ((Labels[:Train]-altr)**2).sum()
stotte = ((Labels[Train:]-alte)**2).sum()
sreg = 0
W = w_g(Fisa[0])
amse = 0
tamse = 0
acc = 0
sregtr = 0
Test = len(db) - Train
MAE = 0
MAPE = 0
smse = 0
# Writer = SummaryWriter("water_loss8")

for epoch in range(Max_epoch):
    for i in range(Train):
        final_out = f_o(Fisa[i], W).sum()
        mse = (final_out - Labels[i]) ** 2
        smse += mse
        MAE += abs(final_out - Labels[i])
        MAPE += abs(final_out - Labels[i])/Labels[i]
        mse.backward()
        W.data -= Lr * W.grad.data
        torch.zero_(W.grad.data)
    r2tr = 1 - (smse / stottr)
    rmse = sqrt(smse/Train)
    MAE /= Train
    MAPE = MAPE / Train * 100
    # Writer.add_scalar("RMSE_train_869", amse, epoch)
    # Writer.add_scalar("R2_train_869", r2tr, epoch)
    print('ES_train: RMSE={}, R2={}, MAE={}, MAPE={}'.format(rmse, r2tr, MAE, MAPE))
    smse = 0
    MAE = 0
    MAPE = 0
    if (epoch+1) % 10 == 0:
        for i in range(Train, len(db)):
            fout = f_o(Fisa[i], W).sum()
            smse += (fout - Labels[i]) ** 2
            MAE += abs(fout - Labels[i])
            MAPE += abs(fout - Labels[i]) / Labels[i]
        r2te = 1 - (smse / stotte)
        rmse = sqrt(smse / Test)
        MAE /= Test
        MAPE = MAPE / Test * 100
        print('Epoch:{},ES_test: RMSE={}, R2={}'.format(epoch, rmse, r2te))
        print('Epoch:{},ES_test: MAE={}, MAPE={}'.format(epoch, MAE, MAPE))
        smse = 0
        MAE = 0
        MAPE = 0
        # Writer.add_scalar("RMSE_test_217", tamse, epoch)
        # Writer.add_scalar("R2_test_217", r2, epoch)
# Writer.close()

