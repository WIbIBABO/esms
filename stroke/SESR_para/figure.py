import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# 1  82  1  1  1  1  1  271.73999023  92    1
# 0  10  0  0  0  0  0  55.11999893   11.5  0
# 1  72  1  1  1  1  1  216.6199913   80.5  1

para = np.loadtxt('./w11.csv', dtype=np.float32, encoding='utf-8').reshape(10, -1)
b = ''
for i in range(10):
    a = ''
    for j in range(10):
        a += '+' + str(para[i][j]) + '*(2*(Y-10)/72-1)**' + str(j)
    a = '(-1)**' + str(i) + '*' + '(' + a + ')'
    b += '+' + a

c = ''
for i in range(10):
    a = ''
    for j in range(10):
        a += '+' + str(para[i][j]) + '*(2*(Y-10)/72-1)**' + str(j)
    a = '(1)**' + str(i) + '*' + '(' + a + ')'
    c += '+' + a

fig = plt.figure()
ax3 = plt.axes(projection='3d')
plt.xlabel("Female                      Male", fontsize = 15)
plt.ylabel("Age", fontsize=15)
plt.xticks(alpha=0)
plt.tick_params(axis='x', width=0)
x1 = np.arange(-0.5, 0.5, 0.01)
y1 = np.arange(10, 82, 0.1)
X, Y = np.meshgrid(x1, y1)
Z1 = eval(b)
ax3.plot_surface(X, Y, Z1, cmap='viridis_r')

x2 = np.arange(0.5, 1.5, 0.01)
y2 = np.arange(10, 82, 0.1)
X, Y = np.meshgrid(x2, y2)
Z2 = eval(c)
ax3.plot_surface(X, Y, Z2, cmap='viridis_r')
plt.show()


# 0  0  0.530924769103
# 1  0  0.734661661023
# 0  1  0.753958156303
# 1  1  0.730294158223
# ax = plt.subplot(projection='3d')  # 三维图形
# plt.xlabel("0                      1", fontsize = 15)
# plt.ylabel("1                      0", fontsize=15)
# ax.set_zlabel('Value')
# plt.xticks(alpha=0)
# plt.tick_params(axis='x', width=0)
# plt.yticks(alpha=0)
# plt.tick_params(axis='y', width=0)
# ax.bar3d(0, 0, 0, dx=1, dy=1, dz=0.5309, color='lightgreen')
# ax.bar3d(1, 0, 0, dx=1, dy=1, dz=0.7346, color='yellowgreen')
# ax.bar3d(0, 1, 0, dx=1, dy=1, dz=0.7539, color='#87CEFA')
# ax.bar3d(1, 1, 0, dx=1, dy=1, dz=0.7302, color='turquoise')
# plt.title('Hypertension and Heart Disease')
# plt.show()


# 0     0     0.574041171674
# 0     1     0.303810349598
# 0.25  0     0.88255984578
# 0.25  1     0.84947381651
# 0.5   0     0.9466809825
# 0.5   1     0.9293643355
# 0.75  0     0.994564956030
# 0.75  1     1.0024633683236
# 1     0     1.002855850
# 1     1     1.027362558802
# ax = plt.subplot(projection='3d')  # 三维图形
# plt.xlabel("0     0.25     0.5    0.75     1", fontsize = 15)
# plt.ylabel("Urban                      Rural", fontsize=15)
# ax.set_zlabel('Value')
# plt.xticks(alpha=0)
# plt.tick_params(axis='x', width=0)
# plt.yticks(alpha=0)
# plt.tick_params(axis='y', width=0)
# ax.bar3d(0, 0, 0, dx=1, dy=1, dz=0.5740, color='lime')
# ax.bar3d(0, 1, 0, dx=1, dy=1, dz=0.3038, color='yellow')
# ax.bar3d(1, 0, 0, dx=1, dy=1, dz=0.8825, color='lightgreen')
# ax.bar3d(1, 1, 0, dx=1, dy=1, dz=0.8494, color='orange')
# ax.bar3d(2, 0, 0, dx=1, dy=1, dz=0.9466, color='yellowgreen')
# ax.bar3d(2, 1, 0, dx=1, dy=1, dz=0.9293, color='tomato')
# ax.bar3d(3, 0, 0, dx=1, dy=1, dz=0.9945, color='turquoise')
# ax.bar3d(3, 1, 0, dx=1, dy=1, dz=1.0024, color='pink')
# ax.bar3d(4, 0, 0, dx=1, dy=1, dz=1.0028, color='#87CEFA')
# ax.bar3d(4, 1, 0, dx=1, dy=1, dz=1.0273, color='peachpuff')
# plt.title('Hypertension and Heart Disease')
# plt.show()





# fig = plt.figure(figsize=(10,6))  #定义新的三维坐标轴
# ax3 = plt.axes(projection='3d')
# plt.xlabel("Diabetes Pedigree Function")
# plt.ylabel("Age")
# xx = np.arange(0.078, 2.42, 0.02)
# yy = np.arange(21, 81, 0.1)
# X, Y = np.meshgrid(xx, yy)
# Z = eval(b)
# ax3.plot_surface(X, Y, Z, cmap='rainbow')
# #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
# plt.show()



# 1  82  1  1  1  1  1  271.73999023  92    1
# 0  10  0  0  0  0  0  55.11999893   11.5  0
# 1  72  1  1  1  1  1  216.6199913   80.5  1
# para = np.loadtxt('./w444.csv', dtype=np.float32, encoding='utf-8').reshape(10, -1)
# b = ''
# for i in range(10):
#     a = ''
#     for j in range(10):
#         a += '+' + str(para[i][j]) + '*(2*(Y-11.5)/80.5-1)**' + str(j)
#     a = '(2*(X-55.12)/216.62-1)**' + str(i) + '*' + '(' + a + ')'
#     b += '+' + a
# print(b)
# fig = plt.figure()  #定义新的三维坐标轴
# ax3 = plt.axes(projection='3d')
# plt.xlabel("Avg Glucose Level", fontsize=15)
# plt.ylabel("BMI", fontsize=15)
# #定义三维数据
# xx = np.arange(55.12, 271.74, 0.5)
# yy = np.arange(11.5, 92, 0.2)
# X, Y = np.meshgrid(xx, yy)
# Z = eval(b)
# ax3.plot_surface(X, Y, Z, cmap='rainbow')
# #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
# plt.show()





# 0    0      0.88467335644
# 0  0.5      0.75816809499
# 0    1      0.7109743117200003
# 1    0      0.8596277093
# 1  0.5      0.981781189
# 1    1      1.01862813898

# ax = plt.subplot(projection='3d')  # 三维图形
# plt.xlabel("0                      1", fontsize = 15)
# plt.ylabel("1             0.5             0", fontsize=15)
# # ax.set_zlabel('Value')
# plt.xticks(alpha=0)
# plt.tick_params(axis='x', width=0)
# plt.yticks(alpha=0)
# plt.tick_params(axis='y', width=0)
# ax.bar3d(0, 0, 0, dx=1, dy=1, dz=0.8846, color='yellowgreen')
# ax.bar3d(0, 1, 0, dx=1, dy=1, dz=0.7581, color='lightgreen')
# ax.bar3d(0, 2, 0, dx=1, dy=1, dz=0.7109, color='turquoise')
# ax.bar3d(1, 0, 0, dx=1, dy=1, dz=0.8596, color='yellow')
# ax.bar3d(1, 1, 0, dx=1, dy=1, dz=0.9466, color='orange')
# ax.bar3d(1, 2, 0, dx=1, dy=1, dz=1.0186, color='tomato')
# # ax.bar3d(3, 0, 0, dx=1, dy=1, dz=0.9945, color='turquoise')
# # ax.bar3d(3, 1, 0, dx=1, dy=1, dz=1.0024, color='pink')
# # ax.bar3d(4, 0, 0, dx=1, dy=1, dz=1.0028, color='#87CEFA')
# # ax.bar3d(4, 1, 0, dx=1, dy=1, dz=1.0273, color='peachpuff')
# plt.title('Hypertension and Heart Disease')
# plt.show()