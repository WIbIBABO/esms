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


def b_s(od, var):
    x = torch.tensor([var], dtype=torch.float32)
    y = torch.tensor([var], dtype=torch.float32)
    bs = torch.tensor([1], dtype=torch.float32)

    for i in range(0, od):
        bs = torch.cat([bs, x], -1)
        x = x * y

    return bs


def f_s(od_sqc, inp_sqc):
    fs = b_s(od_sqc, inp_sqc[0]).reshape(-1, 1)

    for i in range(1, len(inp_sqc)):
        cs = b_s(od_sqc, inp_sqc[i])
        fs = (fs * cs).reshape(-1, 1)

    fs = torch.reshape(fs, [1, -1])

    return fs


def f_s_a(inp_a, od, lots):
    fsa = f_s(od, inp_a[0])
    # fsa = torch.zeros([lots, len(fs)])

    for i in range(1, lots):
        cfs = f_s(od, inp_a[i])
        fsa = torch.cat((fsa, cfs), 0)
        # fsa[i] = cfs

    return fsa


def obsv(fsa, len_t):
    la = len(fsa)
    lat = len_t
    tfsa = fsa[:len_t, 0:]
    lfs = len(fsa[0])
    av = torch.randn([lfs])
    av1 = torch.randn([lfs])
    fsat = fsa.t()

    for i in range(lfs):
        k = abs(fsat[i]).sum()
        av[i] = k / la

    tfsat = tfsa.t()

    for i in range(lfs):
        k = abs(tfsat[i]).sum()
        av1[i] = k / lat


    # for i in range(lfs):
    #     for j in range(la):
    #          k += abs(fsa[j][i])
    #
    #     k = k / la
    #     av[i] = k
    #     k = 0

    return av, av1


def ord_deg(av, l_inp, od):
    mo = l_inp * od
    oddg = torch.randn([mo + 1])
    oddg[0] = 1
    s = 0

    for i in range(1, mo + 1):
        dl = d_l(l_inp, od, i)
        if i < mo:
            dl1 = d_l(l_inp, od, i + 1)
            g = len(dl) - len(dl1)
            ga = dl[:g, 0]
            for j in ga:
                s += av[j]
            s /= g
            oddg[i] = s
            s = 0

        else:
            oddg[i] = av[-1]

    noddg = torch.randn([mo + 1])

    for i in range(mo+1):
        for j in range(10000):
            de = oddg[i]*(10**j)
            if abs(de) >= 1:
                noddg[i] = j
                break

    return oddg, noddg


def at_l(len_se, od, noddg, bs_l, ga):
    l = torch.randn([(od + 1)**len_se])
    l[0] = bs_l
    lno = len(noddg)

    for i in range(1, lno):
        dl = d_l(len_se, od, i)
        o = noddg[i] // ga + 1
        a = bs_l * o

        for j in dl:
            l[j] = a

    return l


def w_g(fs):
    w = torch.normal(0, 0.01, [len(fs)], requires_grad=True, dtype=torch.float32)
    # w.retain_grad()

    return w


def f_o(fs, w):
    fo = fs * w

    return fo


def d_l(linp, od, oli):
    o = torch.tensor([1.001], dtype=torch.float64)
    oseq = torch.zeros(linp, dtype=torch.float64)
    li = torch.tensor([oli], dtype=torch.float64)

    for i in range(0, linp):
        oseq[i] = o

    oseri = f_s(od, oseq)
    droli = torch.tensor([o ** li], dtype=torch.float64)
    dl = (oseri >= droli).nonzero()

    if len(dl):
        return dl
    else:
        dl = torch.tensor([(1 + od) ** linp + 1])
        return dl


def h_o_d1(fsa, dl):
    fsa = fsa.cuda()
    dl = dl.cuda()
    a = len(dl)
    b = len(fsa)
    c = len(fsa[0])
    # j = 0

    if dl[0] >= len(fsa[0]):
        fisa = fsa
    else:
        if a > 1:
            fsa = fsa.t()
            # fsa = fsa.numpy()
            fisa = fsa[0]
            # fisa.cuda()

            # for i in range(a):
            #     fsa = np.delete(fsa, dl[i]-j, axis=1)
            #     j += 1

            for i in range(1, c):
                if i not in dl:
                    fisa = torch.cat([fisa, fsa[i]], 0)
                    if i % 50 == 0:
                        print(i)
            fisa = torch.reshape(fisa, [c - a, b])
            fisa = fisa.t()
        else:
            fisa = torch.zeros([b, c - 1])

            for i in range(b):
                fisa[i] = fsa[i][:-1]
                if i % 10 == 0:
                    print(i)
        fisa = fisa.to(torch.device('cpu'))
        # fisa = torch.from_numpy(fsa)

    return fisa


def h_o_d2(fsa, dl):
    a = len(dl)
    b = len(fsa)
    c = len(fsa[0])
    d = dl[0]

    if dl[0] >= len(fsa[0]):
        fisa = fsa
    else:
        if a > 1:
            fisa = torch.zeros([b, c - a])
            # fisa.cuda()

            for i in range(b):
                cs = torch.cat([fsa[i][:d], fsa[i][d + 1:dl[1]]])

                for j in range(1, a - 1):
                    cs = torch.cat((cs, fsa[i][dl[j] + 1:dl[j + 1]]))

                fisa[i] = cs
                if i % 10 == 0:
                    print(i)
        else:
            fisa = torch.zeros([b, c - 1])

            for i in range(b):
                fisa[i] = fsa[i][:-1]
                if i % 10 == 0:
                    print(i)

    return fisa


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


def rf_w(w, asgrad, lr):
    w = w - lr * asgrad

    return w


def be_so(fisa, label):
    fisa = fisa.cuda()
    label = label.cuda()
    # rlabel = torch.reshape(label, [-1, 1])
    A = fisa.t()@fisa
    B = torch.linalg.inv(A)
    s = A@B
    print(s[0])
    C = B@fisa.t()
    print(C.shape)
    D = C@label
    D = torch.reshape(D, [1, -1])
    D = D.to(torch.device('cpu'))

    return D


def stdError_func(y_test, y):
    return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
    return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
    y_mean = np.array(y)
    y_mean[:] = y.mean()
    return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)


def unit_vector(vec):
    s = sqrt((vec * vec).sum())

    vec /= s
    return vec


def mae(y_pre, y):
    ma = (abs(y_pre - y).sum())/len(y)

    return ma


def mape(y_pre, y):
    map = 100 * ((abs(y_pre - y)/y).sum())/len(y)

    return map


torch.set_printoptions(precision=7)

db = numpy.loadtxt('./waterdata.csv', dtype=numpy.float32, encoding='utf-8')
np.random.shuffle(db)

Inputs = db[:, :-1]
Inputs = i_n(Inputs, 1)

Labels = torch.from_numpy(db[:, -1])

Ordernum = 3
deg = 3
Lr = 0.0005
Train = 869
Test = len(db) - Train
Max_epoch = 10000
avermse = 0
averr2 = 0
avpomse = 0
avpor2 = 0

altr = Labels[:Train].sum()/Train
alte = Labels[Train:].sum()/Test
stottr = ((Labels[:Train]-altr)**2).sum()
stotte = ((Labels[Train:]-alte)**2).sum()

Fisa = f_s_a(Inputs, Ordernum, len(db))
print(Fisa.shape)
W = w_g(Fisa[0])
smse = 0
tamse = 0
sregtr = 0
MAE = 0
MAPE = 0


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
    rmse = sqrt(smse/Train)
    r2tr = 1 - (smse / stottr)
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
        rmse = sqrt(smse / Test)
        r2te = 1 - (smse / stotte)
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
