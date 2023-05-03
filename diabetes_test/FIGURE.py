import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
def f1(t):
    y1 = eval(b)
    return y1
def f2(t):
    y2 = 1*t/t
    return y2

# 199         122         846          67.09999847      2.42000008       81
#   0           0           0                    0      0.078            21
# 199         122         846          67.09999847      2.34200007       60
a = np.loadtxt('./720.csv', dtype=np.float32, encoding='utf-8')
b = a.reshape(1, -1)
para = b.reshape(8, -1)[5]
# para = np.around(para, 4)
print(para)
b = ''
# for i in range(8):
#     b += '+' + str(para[i]) + '*(2*(t-21)/60-1)**' + str(i)
# print(b)
# t = np.linspace(21,80,num = 5000)
# y1 = f1(t)
# y2 = f2(t)
# plt.figure(figsize=(10, 6))
# plt.yticks(fontsize=20,color='#000000')
# plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
# plt.plot(t,y1,'r-')
# plt.plot(t,y2,'k--')
# plt.legend(['Age'], fontsize=25)#图例
# plt.xlabel('x5(Age)', fontsize=25)#设置x,y轴标记
# plt.ylabel('U(x5)', fontsize=25)
# plt.ylim([0.4, 1.4])
# plt.show()


# DPF

# for i in range(8):
#     b += '+' + str(para[i]) + '*(2*(t-0.078)/2.342-1)**' + str(i)
# print(b)
# t = np.linspace(0.078,2.420,num = 5000)
# y1 = f1(t)
# y2 = f2(t)
# plt.figure(figsize=(10, 6))
# plt.yticks(fontsize=20,color='#000000')
# plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
# plt.plot(t,y1,'r-')
# plt.plot(t,y2,'k--')
# plt.legend(['Diabetes Pedegree Function'], fontsize=25)#图例
# plt.xlabel('x4(Diabetes Pedegree Function)', fontsize=25)#设置x,y轴标记
# plt.ylabel('U(x4)', fontsize=25)
# plt.ylim([0.3, 1.1])
# plt.show()


# bmi

for i in range(8):
    b += '+' + str(para[i]) + '*(2*(t-0)/67.09-1)**' + str(i)
print(b)
t = np.linspace(0,67.09,num = 5000)
y1 = f1(t)
y2 = f2(t)
plt.figure(figsize=(10, 6))
plt.yticks(fontsize=20,color='#000000')
plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
plt.plot(t,y1,'r-')
plt.plot(t,y2,'k--')
plt.legend(['BMI'], fontsize=25)#图例
plt.xlabel('x3(BMI)', fontsize=25)#设置x,y轴标记
plt.ylabel('U(x3)', fontsize=25)
# plt.ylim([0.5, 1.5])
plt.show()


# insulin

# for i in range(8):
#     b += '+' + str(para[i]) + '*(2*(t-0)/846-1)**' + str(i)
# print(b)
# t = np.linspace(0,846,num = 5000)
# y1 = f1(t)
# y2 = f2(t)
# plt.yticks(fontsize=20,color='#000000')
# plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
# plt.plot(t,y1,'r-')
# plt.plot(t,y2,'k--')
# plt.legend(['Insulin'], fontsize=25)#图例
# plt.xlabel('x2(Insulin)', fontsize=25)#设置x,y轴标记
# plt.ylabel('U(x2)', fontsize=25)
# plt.ylim([0.5, 1.5])
# plt.show()


# blood pressure

# for i in range(8):
#     b += '+' + str(para[i]) + '*(2*(t-0)/122-1)**' + str(i)
# print(b)
# t = np.linspace(0,122,num = 5000)
# y1 = f1(t)
# y2 = f2(t)
# plt.yticks(fontsize=20,color='#000000')
# plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
# plt.plot(t,y1,'r-')
# plt.plot(t,y2,'k--')
# plt.legend(['Blood Pressure'], fontsize=25)#图例
# plt.xlabel('x1(Blood Pressure)', fontsize=25)#设置x,y轴标记
# plt.ylabel('U(x1)', fontsize=25)
# plt.show()


# Glucose

# for i in range(8):
#     b += '+' + str(para[i]) + '*(2*(t-0)/199-1)**' + str(i)
# print(b)
# t = np.linspace(0,199,num = 5000)
# y1 = f1(t)
# y2 = f2(t)
# plt.yticks(fontsize=20,color='#000000')
# plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
# plt.plot(t,y1,'r-')
# plt.plot(t,y2,'k--')
# plt.legend(['Glucose'], fontsize=25)#图例
# plt.xlabel('x0(Glucose)', fontsize=25)#设置x,y轴标记
# plt.ylabel('U(x0)', fontsize=25)
# plt.show()

