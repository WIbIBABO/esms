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


def unit_vector(vec):
    s = sqrt((vec * vec).sum())

    vec /= s
    return vec


torch.set_printoptions(precision=7)
# device = torch.device("cuda:0")

db = numpy.loadtxt('./waterfuture./1daywater.csv', dtype=numpy.float32, encoding='utf-8')
np.random.shuffle(db)

Inputs = db[:, :-1]
Inputs = i_n(Inputs, 1)

Labels = torch.from_numpy(db[:, -1])

Ordernum = 3
deg = 80
Lr = 0.0005
Train = 870
Max_epoch = 10000
avermse = 0
averr2 = 0
avpomse = 0
avpor2 = 0


# lin_reg = linear_model.LinearRegression()
# poly_reg = PolynomialFeatures(degree=deg)
# X_ploy = poly_reg.fit_transform(Inputs)
#
# xtrain = X_ploy[:Train, 0:]
# xtest = X_ploy[Train:, 0:]
# ytrain = Labels[:Train]
# ytest = Labels[Train:]
#
# lin_reg.fit(xtrain, ytrain)
#
# predict_y = lin_reg.predict(xtrain)
# strError = stdError_func(predict_y, ytrain)
# R2_1 = R2_1_func(predict_y, ytrain)
# # score = lin_reg_2.score(xtest, ytest)  ##sklearn中自带的模型评估，与R2_1逻辑相同
# # print("coefficients", lin_reg_2.coef_)
# print('poly_train: mse={}, R2_1={}'.format(strError, R2_1))
#
# predict_y = lin_reg.predict(xtest)
# strError = stdError_func(predict_y, ytest)
# R2_1 = R2_1_func(predict_y, ytest)
# print('poly_test:参数数量:{}, RMSE={}, R2={}'.format(len(X_ploy[0]), strError, R2_1))
# avpomse += strError
# avpor2 += R2_1

altr = Labels[:Train].sum()/Train
alte = Labels[Train:].sum()/(len(db)-Train)
print(alte)
# al1 = Labels.sum()/len(db)
#
stottr = 0
stotte = 0
sreg = 0
# # reg = 0
for i in range(Train):
    stottr += (Labels[i] - altr)**2
for i in range(Train, len(db)):
    stotte += (Labels[i] - alte)**2
print(stotte)
#
# for i in range(len(db)):
#     stot1 += (Labels[i] - al1)**2
#
# print(al)
# print(al1)
# print(stot)
# print(stot1)
# a = len(db)
# inps = numpy.delete(db, -1, axis=1)
# inps = numpy.delete(db, -500, axis=0)
# labs = numpy.zeros(a)
#
# for i in range(a):
#     labs[i] = db[i][-1]
# Inputs, Labels = g_d_f_d('./waterdata.xlsx', 'utf-16')
# Inputs = torch.from_numpy(Inputs)
# Labels = torch.from_numpy(Labels)
# print(Inputs[0])
# A = len(Inputs)
# B = len(Inputs[0])

# Fs = f_s(Ordernum, Inputs[0])

# t1 = time.time()
# L = d_l(7, Ordernum, 15)
# print(L.shape, L[0], L[0][0])
# print(L[:15, 0][0])
# print(L[:15, 0:][0])
# print(len(L), L[:5])
# t2 = time.time()
# print(t2 - t1)
Fisa = f_s_a(Inputs, Ordernum, len(db))
print(Fisa.shape)

# trfisa = Fisa[:Train, 0:]
# tefisa = Fisa[Train:, 0:]
# trlabels = Labels[:Train]
# telabels = Labels[Train:]
#
# lreg = linear_model.LinearRegression()
# lreg.fit(trfisa, trlabels)
#
# pre = lreg.predict(trfisa)
#
# mmse = stdError_func(pre, trlabels)
# R2_1 = R2_1_func(pre, trlabels)
# print('ergodic_train:参数数量:{}, RMSE={}, R2={}'.format('?', mmse, R2_1))
#
#
# pre = lreg.predict(tefisa)
#
# mmse = stdError_func(pre, telabels)
# R2_1 = R2_1_func(pre, telabels)
# print('ergodic_test:参数数量:{}, RMSE={}, R2={}'.format(len(Fisa[0]), mmse, R2_1))
# avermse += mmse
# averr2 +=R2_1
# if timo == 9:
#     print('多项式拟合阶数：{},参数数量：{},RMSE:{},R2:{}'.format(deg, len(X_ploy[0]), avpomse/10, avpor2/10))
#     print('遍历集拟合阶数：{},参数数量：{},RMSE:{},R2:{}'.format(Ordernum, len(Fisa[0]), avermse/10, averr2/10))

# av, avt = obsv(Fisa, Train)
#
# ss, ss1 = ord_deg(av, 7, 3)
# print(ss, '\n', ss1)
# all = at_l(7, 3, ss1, 0.002, 2)
# print(all)
# all1 = all
# all2 = torch.stack([all, all1], 0)
# all = h_o_d2(all2, L)
# all = all[0]
# print(Fisa.t()[-1], max(Fisa.t()[-1]))
# print(av, '\n', avt, '\n', av - avt)
# print(ss)
# print(ss1)
# print(all, len(all))
# # print(Fusa.shape)
# Fisa = h_o_d1(Fusa, L)
# print(Fisa.shape)
# Fisa = h_o_d2(Fusa, L)
# print(Fisa[-20:])

# fisa = High_order_drop(fusa, L)
# Fisa = Fisa.cuda()
# print(len(fisa), len(fisa[0]))
W = w_g(Fisa[0])
# print(W[:5], W[-5:])
# Acc = torch.tensor([0])
# Accu = torch.tensor([0])
# Sgrad = torch.zeros(len(Fisa[0]), dtype=torch.float32)
# a = torch.tensor([1], dtype=torch.float32)
amse = 0
tamse = 0
acc = 0
# print(sys.getsizeof(Fisa))
sregtr = 0
# Writer = SummaryWriter("water_loss8")
# Writerloss = SummaryWriter("trainlog1")

for epoch in range(Max_epoch):
    for i in range(Train):
        final_out = f_o(Fisa[i], W)

        # print(Fisa[i].shape, Fisa[i][:5])
        # print((1/Fisa[i]).shape, (1/Fisa[i])[:5])
        # final_sum = final_out.sum()
        # if epoch % 100 == 0:
        #     print(final_sum, Labels[i])

        # print(final_sum)

        # if abs(final_sum - Labels[i]) < 0.5:
        #     Acc = Acc + 1
        # print(final_sum)
        L = mse(final_out, Labels[i])
        # print(L.is_leaf)
        # print(2*L)
        amse += L
        sregtr += (final_out.sum() - Labels[i]) ** 2
        # if i % 100 == 0:
        # print("第{}轮，损失为{}".format(epoch + 1, l))
        L.backward()
        # print(w.grad.data)
        # Sgrad = Sgrad + W.grad.data
        # W.data -= Lr * W.grad.data
        W.data -= Lr * W.grad.data
        # W.data -= Lr * W.grad.data
        torch.zero_(W.grad.data)
    amse = sqrt(2*amse/Train)
    r2tr = 1 - (sregtr / stottr)
    # Writer.add_scalar("RMSE_train_869", amse, epoch)
    # Writer.add_scalar("R2_train_869", r2tr, epoch)
    print(amse)
    print(r2tr)
    amse = 0
    sregtr = 0
    if (epoch+1) % 10 == 0:
        for i in range(Train, len(db)):
            fout = f_o(Fisa[i], W)
            fsum = fout.sum()
            #
            if (epoch+1) % 100 == 0:
                print(fsum, Labels[i])
            smse = mse(fsum, Labels[i])
            tamse += smse
            sreg += (fsum - Labels[i])**2
            # reg += (fsum - al)**2
        r2 = 1 - (sreg / stotte)
        # r21 = 1 - (sreg / stot1)
        # r21 = reg / stot
        print('第{}轮的测试集R2为{}'.format(epoch, r2))
        # print('第{}轮的测试集r21为{}'.format(epoch, r21))
        sreg = 0
        # reg = 0
        tamse = sqrt(2*tamse/(len(db)-Train))
        print('第{}轮的测试集损失为{}'.format(epoch, tamse))
        # Writer.add_scalar("RMSE_test_217", tamse, epoch)
        # Writer.add_scalar("R2_test_217", r2, epoch)
        # print('ACC:{}'.format(acc/(len(db)-Train)))
        # acc = 0
        tamse = 0
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
