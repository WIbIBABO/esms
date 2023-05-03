from math import sqrt
from tensorboardX import SummaryWriter
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import warnings;warnings.filterwarnings('ignore')


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


def plot_acc(acc):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(acc)), acc, label="ACCURACY ON TEST SET", color='orangered')
    plt.xlabel("EPOCH")
    plt.ylabel("ACCURACY")
    plt.ylim(min(acc), max(acc))
    plt.legend()
    plt.show()


torch.set_printoptions(precision=7)

db = np.loadtxt('./stroke.csv', dtype=np.float32, encoding='utf-8', delimiter=',')
# np.random.shuffle(db)
db = db[np.argsort(db[:, -1])]

db0 = db[:3246, :]
db1 = db[3246:, :]
np.random.shuffle(db0)
np.random.shuffle(db1)
sub1 = db0[:2596, :]
sub2 = db1[:144, :]
trainset = np.vstack((sub1, sub2))

# np.random.shuffle(trainset)
sub1 = db0[2596:, :]
sub2 = db1[144:, :]
testset = np.vstack((sub1, sub2))
db = np.vstack((trainset, testset))

inputs = db[:, :-1]
inputs, maxm, minm, dm = i_n(inputs, 1)
print(maxm, minm, dm)
labels = torch.from_numpy(db[:, -1])
od = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
lr = 0.002
amse = 0
max_epoch = 400
train = 2740
test = len(db) - train
o = f_s_a(inputs, od)
w, b = w_b_g(od)
l = len(inputs)
ws = 0
bs = 0
p = 0
print(labels[:train])
print(labels[train:])
acctrain = 0
acctest = 0

TP = 0
TN = 0
FP = 0
FN = 0
# diaacc = SummaryWriter("dia_acc7")

ACC = []
for epoch in range(max_epoch):
    for i in range(train):
        mul = mulp_sum(w * o[i], od)
        # if epoch % 50 == 0:
        #     print(b * mul.data, labels[i])
        loss = (b * mul - labels[i]) ** 2
        if abs(b * mul-labels[i]) < 0.5:
            acctrain += 1

        # print(loss)
        loss.backward()
        if labels[i] == 1:
            ws += 10*w.grad.data
            bs += 10*b.grad.data
        else:
            ws += w.grad.data
            bs += b.grad.data


        torch.zero_(w.grad.data)
        torch.zero_(b.grad.data)
     #diaacc.add_scalar("acc_train_614", acctrain/train, epoch)
    ws /= train
    bs /= train
    w.data -= lr * unit_vector(ws)
    b.data -= lr * bs
    ws = 0
    bs = 0
    print(acctrain/train)
    acctrain = 0

    for i in range(train, len(db)):
        fout = mulp_sum(w * o[i], od)
        if abs(b * fout-labels[i]) < 0.5:
            acctest += 1
        # else:
        #     print(b * fout,labels[i])

    print('第{}轮的测试集acc为{}'.format(epoch, acctest/test))
    # diaacc.add_scalar("acc_test_154", acctest / (len(db)-train), epoch)
    ACC.append(acctest/test)
    acctest = 0


for i in range(train, len(db)):
    fout = mulp_sum(w * o[i], od)
    if labels[i] == 1:
        if abs(b * fout - labels[i]) < 0.5:
            TP += 1
        else:
            FN += 1
    if labels[i] == 0:
        if abs(b * fout - labels[i]) < 0.5:
            TN += 1
        else:
            FP += 1

TPR = TP/(TP + FN)
TNR = TN/(TN + FP)
FPR = 1-TNR
print('TPR:{}'.format(TPR), 'TNR:{}'.format(TNR), 'FPR:{}'.format(FPR), 'Acc:{}'.format((TP+TN)/(TP+TN+FP+FN)))
plot_acc(ACC)
datapre = np.zeros([len(db)-train])
for i in range(train, len(db)):
    fout = mulp_sum(w * o[i], od)
    datapre[i-train] = b * fout

datalabel = labels[train:]
data = w.data.numpy()
frame = pd.DataFrame(data)
frame1 = pd.DataFrame(datalabel)
frame2 = pd.DataFrame(datapre)
frameb = pd.DataFrame(b.data.numpy())
frame.to_csv("./CESR_para/w.csv", index=False, header=False, sep=' ')
frameb.to_csv("./CESR_para/b.csv", index=False, header=False, sep=' ')
frame1.to_csv("./CESR_para/label.csv", index=False, header=False, sep=' ')
frame2.to_csv("./CESR_para/pre.csv", index=False, header=False, sep=' ')
# diaacc.close()

fpr, tpr, threshold = metrics.roc_curve(labels[-test:], datapre)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(10,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %1f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



