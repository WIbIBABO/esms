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
    fs = f_s(od, inp_a[0])[1:]

    fsa = torch.zeros([lots, len(fs)])

    for i in range(lots):
        cfs = f_s(od, inp_a[i])[1:]
        fsa[i] = cfs
    fsa = torch.reshape(fsa, [1, lots, -1])
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


def fuloss(b, w:dict, fsa, j, label):
    loss = 1
    for i in range(0, 8):
        loss *= 1 + f_o(fsa[i][j], w['w'+str(i)])
    pre = b * loss
    loss = (b * loss-label)**2

    return loss, pre


def s(w:dict,c:dict):
    for x in range(0, 8):
        c["c{}".format(x)] += w['w'+str(x)].grad.data
        torch.zero_(w['w' + str(x)].grad.data)
    return c


def update(w:dict,c:dict, Lr):
    for x in range(0, 8):
        w['w'+str(x)].data -= Lr * unit_vector(c["c{}".format(x)]/ train)
    return w


torch.set_printoptions(precision=7)

Lr = torch.tensor([0.002])
train = 863
max_epoch = torch.tensor([10000])

db = numpy.loadtxt('./1daywater.csv', dtype=numpy.float32, encoding='utf-8')
np.random.shuffle(db)
# Inputs = torch.from_numpy(db[:-1, :])
# print(Inputs.shape)
# Labels = torch.from_numpy(db[1:, -1]).reshape(-1,1)
# print(Labels.shape)
# print(len(Inputs), len(Labels))
# datas = torch.cat([Inputs, Labels], -1)
# print(datas.shape)
# datab = datas.data.numpy()
# frame1 = pd.DataFrame(datab)
# frame1.to_csv("./1daywater.csv", index=False, header=False, sep=' ')

Inputs = db[:, :-1]
Inputs = i_n(Inputs, 1)
Labels = torch.from_numpy(db[:, -1])

altr = Labels[:train].sum()/train
alte = Labels[train:].sum()/(len(db)-train)
stottr = 0
stotte = 0
sreg = 0

for i in range(train):
    stottr += (Labels[i] - altr)**2
for i in range(train, len(db)):
    stotte += (Labels[i] - alte)**2

Fisa = f_s_a(Inputs[:, 0].reshape(-1, 1), 12, len(db))

for i in range(1, 8):
    Fisa = torch.cat([Fisa, f_s_a(Inputs[:, i].reshape(-1, 1), 12, len(db))], 0)


b = torch.rand(1, requires_grad=True)

w={}
for x in range(0, 8):
    w["w{}".format(x)]=w_g(Fisa[0][0])

acc = 0
amse = 0
acctrain = 0
s1 = 0
s2 = 0
s3 = 0
s4 = 0
sb = 0
sregtr = 0

# Writer = SummaryWriter("2d_loss_water")

c={}
for x in range(0, 8):
    c["c{}".format(x)]=0
for epoch in range(max_epoch):
    for i in range(train):

        loss, pre = fuloss(b,w,Fisa,i, Labels[i])

        amse += loss
        loss.backward()
        c = s(w, c)
        sb += b.grad.data

        torch.zero_(b.grad.data)
    print(sqrt(amse / train))
    print(1 - (amse / stottr))
    # Writer.add_scalar("RMSE_train_869", sqrt(amse / train), epoch)
    # Writer.add_scalar("R2_train_869", 1 - (amse / stottr), epoch)
    amse = 0
    w = update(w, c, Lr)
    b.data -= Lr * (sb / train)
    c = {}
    for x in range(0, 8):
        c["c{}".format(x)] = 0
    if (epoch+1) % 10 == 0:
        for i in range(train, len(db)):
            loss, pre = fuloss(b,w,Fisa,i, Labels[i])
            amse += loss
            if (epoch+1) % 10 == 0:
                print(pre, Labels[i])
        r2 = 1 - (amse / stotte)
        print('第{}轮的测试集损失为{}'.format(epoch, sqrt(amse/(len(db)-train))))
        print('第{}轮的测试集R2为{}'.format(epoch, r2))
        # Writer.add_scalar("RMSE_test_217", sqrt(amse/(len(db)-train)), epoch)
        # Writer.add_scalar("R2_test_217", r2, epoch)
        amse = 0
        print(w)
        # print(w['w0'])

# # Writer.close()
# #
# # data1 = W1.data.numpy()
# # data2 = W2.data.numpy()
# # data3 = W3.data.numpy()
# # data4 = W4.data.numpy()
# # datab = b.data.numpy()
# # frame1 = pd.DataFrame(data1)
# # frame2 = pd.DataFrame(data2)
# # frame3 = pd.DataFrame(data3)
# # frame4 = pd.DataFrame(data4)
# # frameb = pd.DataFrame(datab)
# # frame1.to_csv("./data1.csv", index=False, header=False, sep=' ')
# # frame2.to_csv("./data2.csv", index=False, header=False, sep=' ')
# # frame3.to_csv("./data3.csv", index=False, header=False, sep=' ')
# # frame4.to_csv("./data4.csv", index=False, header=False, sep=' ')
# # frameb.to_csv("./datab.csv", index=False, header=False, sep=' ')

