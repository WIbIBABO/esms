import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# [ 34.9000   9.2100   25.6700   1246.       91.1999   16.8999   0.3990   14.0900]
# [  2.2999   7.4800    4.6399   372.8999     4.        2.0999   0.0010    0.6800]
# [ 32.6000   1.7300   21.0300   873.1000    87.1999   14.7999   0.4000   13.4100]
para = np.loadtxt('./results/LESE_para/12order_w6.csv', dtype=np.float32, encoding='utf-8')[6]
# para = np.around(para, 4)
print(para)
b = ''
for i in range(13):
    b += '+' + str(para[i]) + '*(2*(t-0.6800)/13.4100-1)**' + str(i)
print(b)

def f1(t):
    y1 = eval(b)
    return y1
def f2(t):
    y2 = 1*t/t
    return y2

t = np.linspace(0.6800,14.0900,num = 5000)
y1 = f1(t)
y2 = f2(t)
#绘制图像
plt.plot(t,y1,'r-')
plt.plot(t,y2,'k--')
#设置图表属性
plt.legend(['Total Nitrogen'])#图例
plt.xlabel('x6(Total Nitrogen)')#设置x,y轴标记
plt.ylabel('U(x6)')

plt.show()



# fig = plt.figure()  #定义新的三维坐标轴
# ax3 = plt.axes(projection='3d')
# plt.xlabel("Turbidity")
# plt.ylabel("Potassium Permanganate")
# #定义三维数据
# xx = np.arange(-1, 1, 0.01)
# yy = np.arange(-1, 1, 0.01)
# X, Y = np.meshgrid(xx, yy)
# Z = eval(b)
#
#
# ax3.plot_surface(X, Y, Z, cmap='rainbow')
# #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
# plt.show()



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
