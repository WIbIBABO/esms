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
from collections import Counter
import warnings;warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

def plot_acc(acc):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(acc)), acc, label="ACCURACY ON TEST SET", color='orangered')

    plt.xlabel("EPOCH")
    plt.ylabel("ACCURACY")
    plt.ylim(min(acc), max(acc))
    plt.legend()
    plt.show()


torch.set_printoptions(precision=7)

Max_epoch = 500
Lr = torch.tensor([0.01])
Train = 614

db = numpy.loadtxt('./diabete.csv', dtype=numpy.float32, encoding='utf-8')
test = len(db) - Train
# np.random.shuffle(db)
db = db[np.argsort(db[:, -1])]
db0 = db[:500, :]
db1 = db[500:, :]
np.random.shuffle(db0)
np.random.shuffle(db1)
sub1 = db0[:400, :]
sub2 = db1[:214, :]
trainset = np.vstack((sub1, sub2))
np.random.shuffle(trainset)
sub1 = db0[400:, :]
sub2 = db1[214:, :]
testset = np.vstack((sub1, sub2))
db = np.vstack((trainset, testset))

Inputs = numpy.delete(db, -1, axis=1)
Inputs = i_n(Inputs, 1)
Labels = numpy.zeros(len(db), dtype=numpy.float32)
for i in range(len(db)):
    Labels[i] = db[i][-1]

Labels = torch.from_numpy(Labels)
Fisa1 = f_s_a(Inputs[:, (1, 5)], 7, len(db))
Fisa2 = f_s_a(Inputs[:, (6, 7)], 7, len(db))
Fisa3 = f_s_a(Inputs[:, (2, 4)], 7, len(db))

b = torch.rand(1, requires_grad=True)
W1 = w_g(Fisa1[0])
W2 = w_g(Fisa2[0])
W3 = w_g(Fisa3[0])

acc = 0
tamse = 0
acctrain = 0

s1 = 0
s2 = 0
s3 = 0
sb = 0
ACC = []
# Writer = SummaryWriter("2d_loss_diabetes1")

for epoch in range(Max_epoch):
    for i in range(Train):
        final_out1 = f_o(Fisa1[i], W1)
        final_out2 = f_o(Fisa2[i], W2)
        final_out3 = f_o(Fisa3[i], W3)
        L = mse(b*(1+final_out1.sum())*(1+final_out2.sum())*(1+final_out3.sum()), Labels[i])
        if abs(b*(1+final_out1.sum())*(1+final_out2.sum())*(1+final_out3.sum()) - Labels[i]) < 0.5:
            acctrain += 1
        L.backward()
        s1 += W1.grad.data
        s2 += W2.grad.data
        s3 += W3.grad.data
        sb += b.grad.data
        torch.zero_(W1.grad.data)
        torch.zero_(W2.grad.data)
        torch.zero_(W3.grad.data)
        torch.zero_(b.grad.data)
    print("trainacc:{}".format(acctrain/Train))
    # Writer.add_scalar("acc_train_614", acctrain/Train, epoch)
    # W1.data -= Lr * unit_vector(s1/Train)
    # W2.data -= Lr * unit_vector(s2/Train)
    # W3.data -= Lr * unit_vector(s3/Train)
    W1.data -= Lr * (s1 / Train)
    W2.data -= Lr * (s2 / Train)
    W3.data -= Lr * (s3 / Train)
    b.data -= Lr * (sb / Train)
    s1 = 0
    s2 = 0
    s3 = 0
    sb = 0
    acctrain = 0
    for i in range(Train, len(db)):
        fout1 = f_o(Fisa1[i], W1).sum()
        fout2 = f_o(Fisa2[i], W2).sum()
        fout3 = f_o(Fisa3[i], W3).sum()
        fsum = b*(1+fout1)*(1+fout2)*(1+fout3)

        if abs(fsum-Labels[i]) < 0.5:
            acc += 1

    acc /= test
    print("epoch:{},testacc:{}".format(epoch, acc))
    # Writer.add_scalar("acc_test_154", acc, epoch)
    ACC.append(acc)
    acc = 0
# Writer.close()
plot_acc(ACC)

data1 = W1.data.numpy()
data2 = W2.data.numpy()
data3 = W3.data.numpy()
datab = b.data.numpy()
frame1 = pd.DataFrame(data1)
frame2 = pd.DataFrame(data2)
frame3 = pd.DataFrame(data3)
frameb = pd.DataFrame(datab)
frame1.to_csv("./SESR_para/w1.csv", index=False, header=False, sep=' ')
frame2.to_csv("./SESR_para/w2.csv", index=False, header=False, sep=' ')
frame3.to_csv("./SESR_para/w3.csv", index=False, header=False, sep=' ')
frameb.to_csv("./SESR_para/b.csv", index=False, header=False, sep=' ')


datapre = np.zeros([154])
for i in range(Train, len(db)):
    fout1 = f_o(Fisa1[i], W1).sum()
    fout2 = f_o(Fisa2[i], W2).sum()
    fout3 = f_o(Fisa3[i], W3).sum()
    fout = b*(1+fout1)*(1+fout2)*(1+fout3)
    datapre[i-Train]= b * fout

datalabel = Labels[Train:]
fpr, tpr, threshold = metrics.roc_curve(Labels[-test:], datapre)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(10,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %1f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
