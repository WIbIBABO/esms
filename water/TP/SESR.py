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

Lr = torch.tensor([0.002])
train = 869
max_epoch = torch.tensor([10000])

db = numpy.loadtxt('./TP_pre.csv', dtype=numpy.float32, encoding='utf-8')
np.random.shuffle(db)

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

Fisa1 = f_s_a(Inputs[:, (0, 1)], 12, len(db))
Fisa2 = f_s_a(Inputs[:, (2, 3)], 12, len(db))
Fisa3 = f_s_a(Inputs[:, (4, 5)], 12, len(db))
Fisa4 = f_s_a(Inputs[:, -1].reshape(-1, 1), 12, len(db))


b = torch.rand(1, requires_grad=True)
W1 = w_g(Fisa1[0])
W2 = w_g(Fisa2[0])
W3 = w_g(Fisa3[0])
W4 = w_g(Fisa4[0])

acc = 0
amse = 0
acctrain = 0
s1 = 0
s2 = 0
s3 = 0
s4 = 0
sb = 0
sregtr = 0

Writer = SummaryWriter("2d_loss_water")

for epoch in range(max_epoch):
    for i in range(train):
        final_out1 = f_o(Fisa1[i], W1)
        final_out2 = f_o(Fisa2[i], W2)
        final_out3 = f_o(Fisa3[i], W3)
        final_out4 = f_o(Fisa4[i], W4)
        loss = (b * (1 + final_out1.sum()) * (1 + final_out2.sum()) * (1 + final_out3.sum()) * (1 + final_out4.sum()) - Labels[i])**2
        amse += loss
        loss.backward()
        s1 += W1.grad.data
        s2 += W2.grad.data
        s3 += W3.grad.data
        s4 += W4.grad.data
        sb += b.grad.data
        torch.zero_(W1.grad.data)
        torch.zero_(W2.grad.data)
        torch.zero_(W3.grad.data)
        torch.zero_(W4.grad.data)
        torch.zero_(b.grad.data)
    print(sqrt(amse / train))
    print(1 - (amse / stottr))
    Writer.add_scalar("RMSE_train_869", sqrt(amse / train), epoch)
    Writer.add_scalar("R2_train_869", 1 - (amse / stottr), epoch)
    amse = 0
    W1.data -= Lr * unit_vector(s1 / train)
    W2.data -= Lr * unit_vector(s2 / train)
    W3.data -= Lr * unit_vector(s3 / train)
    W4.data -= Lr * unit_vector(s4 / train)
    b.data -= Lr * (sb / train)
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    sb = 0
    if (epoch+1) % 10 == 0:
        for i in range(train, len(db)):
            final_out1 = f_o(Fisa1[i], W1)
            final_out2 = f_o(Fisa2[i], W2)
            final_out3 = f_o(Fisa3[i], W3)
            final_out4 = f_o(Fisa4[i], W4)
            loss = (b * (1 + final_out1.sum()) * (1 + final_out2.sum()) * (1 + final_out3.sum()) * (
                        1 + final_out4.sum()) - Labels[i]) ** 2
            amse += loss
            if (epoch+1) % 100 == 0:
                print(b * (1 + final_out1.sum()) * (1 + final_out2.sum()) * (1 + final_out3.sum()) * (
                        1 + final_out4.sum()), Labels[i])
        r2 = 1 - (amse / stotte)
        print('第{}轮的测试集损失为{}'.format(epoch, sqrt(amse/(len(db)-train))))
        print('第{}轮的测试集R2为{}'.format(epoch, r2))
        Writer.add_scalar("RMSE_test_217", sqrt(amse/(len(db)-train)), epoch)
        Writer.add_scalar("R2_test_217", r2, epoch)
        amse = 0
Writer.close()

data1 = W1.data.numpy()
data2 = W2.data.numpy()
data3 = W3.data.numpy()
data4 = W4.data.numpy()
datab = b.data.numpy()
frame1 = pd.DataFrame(data1)
frame2 = pd.DataFrame(data2)
frame3 = pd.DataFrame(data3)
frame4 = pd.DataFrame(data4)
frameb = pd.DataFrame(datab)
frame1.to_csv("./data1.csv", index=False, header=False, sep=' ')
frame2.to_csv("./data2.csv", index=False, header=False, sep=' ')
frame3.to_csv("./data3.csv", index=False, header=False, sep=' ')
frame4.to_csv("./data4.csv", index=False, header=False, sep=' ')
frameb.to_csv("./datab.csv", index=False, header=False, sep=' ')

