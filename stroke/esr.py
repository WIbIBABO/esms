import random
import time
import sys
from math import sqrt
import numpy as np
import pandas as pd
import torch
import numpy
import xlrd as xlrd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from tensorboardX import SummaryWriter
from sklearn import metrics

def b_s(od, var):
    x = torch.tensor([var], dtype=torch.float32)
    y = torch.tensor([var], dtype=torch.float32)
    bs = torch.tensor([1], dtype=torch.float32)

    for i in range(0, od):
        bs = torch.cat([bs, x], -1)
        x = x * y

    return bs


def f_s(od, sqc):
    fs = b_s(od, sqc[0])
    fs = torch.reshape(fs, [len(fs), 1])

    for i in range(1, len(sqc)):
        cs = b_s(od, sqc[i])
        fs = fs * cs
        fs = torch.reshape(fs, [-1, 1])

    fs = torch.reshape(fs, [-1, ])

    return fs


def f_s_a(inp_a, od, lots):
    fs = f_s(od, inp_a[0])
    fsa = torch.zeros([lots, len(fs)])

    for i in range(lots):
        cfs = f_s(od, inp_a[i])
        fsa[i] = cfs

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


# def Get_data_from_net_layer():


def mse(fi_o, label):
    sum = fi_o.sum()
    l = 0.5 * (sum - label) ** 2

    return l


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


torch.set_printoptions(precision=7)

# device = torch.device("cuda:0")
Ordernum = torch.tensor([1])

torch.set_printoptions(precision=7)

Max_epoch = 300
Lr = torch.tensor([0.00001])
Train = 2740

db = numpy.loadtxt('./stroke.csv', dtype=numpy.float32, encoding='utf-8', delimiter=',')
test = len(db) - Train
# np.random.shuffle(db)
db = db[np.argsort(db[:, -1])]
db0 = db[:3246, :]
db1 = db[3246:, :]
np.random.shuffle(db0)
np.random.shuffle(db1)
sub1 = db0[:2596, :]
sub2 = db1[:144, :]
trainset = np.vstack((sub1, sub2))
np.random.shuffle(trainset)
sub1 = db0[2596:, :]
sub2 = db1[144:, :]
testset = np.vstack((sub1, sub2))
db = np.vstack((trainset, testset))

Inputs = numpy.delete(db, -1, axis=1)
Inputs = i_n(Inputs, 1)
Labels = numpy.zeros(len(db), dtype=numpy.float32)
for i in range(len(db)):
    Labels[i] = db[i][-1]


Fisa = f_s_a(Inputs, Ordernum, len(db))


W = w_g(Fisa[0])

acc = 0
tamse = 0
acctrain = 0


for epoch in range(Max_epoch):
    for i in range(Train):
        final_out = f_o(Fisa[i], W)
        L = mse(final_out, Labels[i])
        if abs(final_out.sum() - Labels[i]) < 0.5:
            acctrain += 1
        # if (epoch+1) % 10 == 0:
        #     print(final_out.sum(), Labels[i])
        L.backward()
        if Labels[i] == 1:
            W.data -= Lr * 15 * W.grad.data
        else:
            W.data -= Lr * W.grad.data

        torch.zero_(W.grad.data)
    print("trainacc:{}".format(acctrain/Train))
    acctrain = 0
    if (epoch+1) % 5 == 0:
        for i in range(Train, len(db)):
            fout = f_o(Fisa[i], W)
            fsum = fout.sum()
            if (epoch+1) % 20 == 0:
                print(fsum.data, Labels[i])
            if abs(fsum-Labels[i]) < 0.5:
                acc += 1

        acc /= len(db)-Train
        print(acc,epoch)
        acc = 0
        # print(W[:5], W[-5:])
    # Accu = Acc / Train
    # print(Accu.item())
    # Asgrad = Sgrad.sum() / Train
    # Writer.add_scalar("test", accu, epoch)
    # Writerloss.add_scalar("test", Asgrad, epoch)
    # Acc = 0
    # Accu = 0
    # torch.zero_(Sgrad)
    # torch.zero_(Asgrad)
    # if epoch % 100 ==0:
    #     print("当前是第{}轮".format(epoch))
    #
    # if epoch > 2000:
    #     lr = 0.007

# Writer.close()


#     accu = acc / train
#     print(accu)
#     writer.add_scalar("test", accu, epoch)
#     acc = 0
#     accu = 0
#     print(sgrad.sum())
#     asgrad = sgrad / train
#     # print(asgrad)
#     w.data -= lr * asgrad.data
#     torch.zero_(asgrad)
#     torch.zero_(sgrad)
# writer.close()
datapre = np.zeros([len(db)-Train])
for i in range(Train, len(db)):
    datapre[i-Train] = f_o(Fisa[i], W).sum()

datalabel = Labels[Train:]
fpr, tpr, threshold = metrics.roc_curve(Labels[-test:], datapre)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label='Val AUC = %1f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()