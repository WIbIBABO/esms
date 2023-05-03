import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
para = np.loadtxt('./para/data4.csv', dtype=np.float32, encoding='utf-8').reshape(13, -1)
para = np.around(para, 3)
b = ''
for i in range(13):
    a = ''
    for j in range(13):
        a += '+' + str(para[i][j]) + '*Y**' + str(j)
    a = 'X**' + str(i) + '*' + '(' + a + ')'
    b += '+' + a



print(b)
fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')
plt.xlabel("Turbidity")
plt.ylabel("Potassium Permanganate")
#定义三维数据
xx = np.arange(-1, 1, 0.01)
yy = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(xx, yy)
Z = eval(b)


ax3.plot_surface(X, Y, Z, cmap='rainbow')
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
plt.show()



# a = ''
# for j in range(13):
#     a += '+' + str(para[j][0]) + '*X**' + str(j)
#
# # fig = plt.figure()  #定义新的三维坐标轴
# #
# # plt.xlabel("Total Phosphorus")
# # #定义三维数据
# # xx = np.arange(-1, 1, 0.01)
# # X= np.meshgrid(xx)
# # Z = eval(a)
# # plt.plot(X, Z)
# # #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
# # plt.show()
# print(a)
# X = np.arange(-1, 1, 0.01)
# y= eval(a)
# plt.plot(X, y)
# plt.xlabel("Total Phosphorus")
# plt.ylim(0.8, 1.3)
# plt.legend()
# plt.show()
