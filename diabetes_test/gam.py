import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from pygam import PoissonGAM, s, te, f, l, LinearGAM, LogisticGAM
import pygam
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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


db = numpy.loadtxt('./diabete.csv', dtype=numpy.float32, encoding='utf-8')
np.random.shuffle(db)
Inputs = db[:, (1,2,4,5,6,7)]
Inputs = i_n(Inputs, 1)
Labels = torch.from_numpy(db[:, -1]).reshape(-1, 1)

# gam = PoissonGAM(s(0, n_splines=200, lam=1, spline_order=12) + te(2, 1, lam=1) + te(3, 4, lam=1) + te(5, 6, lam=1)).fit(Inputs, Labels)
gam = LogisticGAM(s(n_splines=8,feature=0,spline_order=7)+s(n_splines=8,feature=1,spline_order=7)+
                 s(n_splines=8,feature=2,spline_order=7)+s(n_splines=8,feature=3,spline_order=7)+
                 s(n_splines=8,feature=4,spline_order=7)+s(n_splines=8,feature=5,spline_order=7)).fit(Inputs[:614, :], Labels[:614, :])
# gam.predict()
# fig, axs = plt.subplots(1, 3)
# titles = ['student', 'balance', 'income']
#
# for i, ax in enumerate(axs):
#     XX = gam.generate_X_grid(term=i)
#     pdep, confi = gam.partial_dependence(term=i, width=.95)
#
#     ax.plot(XX[:, i], pdep)
#     ax.plot(XX[:, i], confi, c='r', ls='--')
#     ax.set_title(titles[i])
#     ax.show()
# gam = LinearGAM().fit(Inputs[:870], Labels[:870])
# gam = LinearGAM(spline_order=4, n_splines=5).fit(Inputs, Labels)
# gam.score(Inputs[:870], Labels[:870])
# gam.summary()
print(gam.accuracy(Inputs[:614, :], Labels[:614, :]))
print(gam.accuracy(Inputs[614:, :], Labels[614:, :]))





# b= gam.predict(Inputs[:870], Labels[:870])
# print(b[:30])
# l=gam.predict(Inputs[870:], Labels[870:])
# print(l)
# from patsy import dmatrix
#
# import statsmodels.api as sm
#
# import statsmodels.formula.api as smf
#
# #生成一个三节点的三次样条（25,40,60）
#
# transformed_x = dmatrix("bs(train, knots=(25,40,60), degree=3, include_intercept=False)", {"train": train_x},return_type='dataframe')
# pygam.s()