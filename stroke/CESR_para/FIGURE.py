import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def f1():
    y1 = eval(b)
    return y1


def f2(t):
    y2 = 1*t/t
    return y2

# 1  82  1  1  1  1  1  271.73999023  92    1
# 0  10  0  0  0  0  0  55.11999893   11.5  0
# 1  72  1  1  1  1  1  216.6199913   80.5  1


a = np.loadtxt('./w.csv', dtype=np.float32, encoding='utf-8')
b = a.reshape(1, -1)
para = b.reshape(10, -1)[0]
# para = np.around(para, 4)
print(para)
b = ''

# 0    0.81372023667
# 1    0.79211812133
for i in range(10):
    b += '+' + str(para[i]) + '*(2*(21-21)/60-1)**' + str(i)
print(eval(b))
# t = np.linspace(-0.495, 1.495, num=500)
#
# y = [0.8137 if (i < 0.5) else 0.7921 for i in t]
# plt.figure(figsize=(9, 6))
# plt.yticks(fontsize=20, color='#000000')
# plt.xticks(fontsize=20, color='#000000')
# plt.plot(t, y, 'r-')
# plt.legend(['Gender'], fontsize=20)
# plt.ylabel('Value', fontsize=20)
# plt.ylim([0.75, 0.83])
#
# cell = ['Female', 'Male']
# pvalue = [0.8137, 0.7921]
#
# width = 1
# index = np.arange(len(cell))
# figsize = (9, 6)  # 调整绘制图片的比例
#
#
# plt.bar('Female', 0.8137, width, color="#87CEFA",)  # 绘制柱状图
# plt.bar('Male', 0.7921, width=width, facecolor='yellowgreen')
# # plt.xlabel('cell type') #x轴
# plt.ylabel('Value', fontsize=20)
# plt.xticks(index, cell, fontsize=20)  # 将横坐标用cell替换,fontsize用来调整字体的大小
# plt.text(0, 0.82, '0.8137', ha='center', va='top',fontsize=20)
# plt.text(1, 0.7984, '0.7921', ha='center', va='top',fontsize=20)
# plt.show()



# for i in range(10):
#     b += '+' + str(para[i]) + '*(2*(t-10)/72-1)**' + str(i)
# print(b)
# t = np.linspace(10, 82, num=5000)
# y1 = f1()
# y2 = f2(t)
# plt.figure(figsize=(9, 6))
# plt.yticks(fontsize=20,color='#000000')
# plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
# plt.plot(t,y1,'r-')
# plt.plot(t,y2,'k--')
# plt.legend(['Age'], fontsize=20)#图例
# # plt.xlabel('Age', fontsize=20)#设置x,y轴标记
# plt.ylabel('Value', fontsize=20)
# # plt.ylim([0.3, 1.1])
# plt.show()


# bmi

# 0    0.7084661959999998
# 1    0.91772387
# for i in range(10):
#     b += '+' + str(para[i]) + '*(2*(81-21)/60-1)**' + str(i)
#
# # print(eval(b))
# t = np.linspace(-0.495, 1.495, num=500)
#
# y = [0.7084 if (i < 0.5) else 0.9177 for i in t]
# plt.figure(figsize=(9, 6))
# plt.yticks(fontsize=20, color='#000000')
# plt.xticks(fontsize=20, color='#000000')
# plt.plot(t, y, 'r-')
# plt.legend(['Hypertension'], fontsize=20)
# plt.ylabel('Value', fontsize=20)
# plt.ylim([0.65, 0.94])
#
# cell = ['0', '1']
# pvalue = [0.8137, 0.7921]
#
# width = 1
# index = np.arange(len(cell))
# figsize = (9, 6)  # 调整绘制图片的比例
#
#
# plt.bar('0', 0.7084, width, color="#87CEFA",)  # 绘制柱状图
# plt.bar('1', 0.9177, width=width, facecolor='yellowgreen')
# # plt.xlabel('cell type') #x轴
# plt.ylabel('Value', fontsize=20)
# plt.xticks(index, cell, fontsize=20)  # 将横坐标用cell替换,fontsize用来调整字体的大小
# plt.text(0, 0.725, '0.7084', ha='center', va='top', fontsize=20)
# plt.text(1, 0.935, '0.9177', ha='center', va='top', fontsize=20)
# plt.show()


# 0.7560
# 0.9351
# for i in range(10):
#     b += '+' + str(para[i]) + '*(2*(21-21)/60-1)**' + str(i)
#
# # print(eval(b))
# t = np.linspace(-0.495, 1.495, num=500)
#
# y = [0.7560 if (i < 0.5) else 0.9351 for i in t]
# plt.figure(figsize=(9, 6))
# plt.yticks(fontsize=20, color='#000000')
# plt.xticks(fontsize=20, color='#000000')
# plt.plot(t, y, 'r-')
# plt.legend(['Heart Disease'], fontsize=20)
# plt.ylabel('Value', fontsize=20)
# plt.ylim([0.65, 0.97])
#
# cell = ['0', '1']
# # pvalue = [0.8137, 0.7921]
#
# width = 1
# index = np.arange(len(cell))
# figsize = (9, 6)  # 调整绘制图片的比例
#
#
# plt.bar('0', 0.7560, width, color="#87CEFA",)  # 绘制柱状图
# plt.bar('1', 0.9351, width=width, facecolor='yellowgreen')
# # plt.xlabel('cell type') #x轴
# plt.ylabel('Value', fontsize=20)
# plt.xticks(index, cell, fontsize=20)  # 将横坐标用cell替换,fontsize用来调整字体的大小
# plt.text(0, 0.774, '0.7560', ha='center', va='top', fontsize=20)
# plt.text(1, 0.953, '0.9351', ha='center', va='top', fontsize=20)
# plt.show()


# 0.8117
# 0.8552
# for i in range(10):
#     b += '+' + str(para[i]) + '*(2*(81-21)/60-1)**' + str(i)
#
# # rint(eval(b))
# t = np.linspace(-0.495, 1.495, num=500)
#
# y = [0.8117 if (i < 0.5) else 0.8552 for i in t]
# plt.figure(figsize=(9, 6))
# plt.yticks(fontsize=20, color='#000000')
# plt.xticks(fontsize=20, color='#000000')
# plt.plot(t, y, 'r-')
# plt.legend(['Ever Married'], fontsize=20)
# plt.ylabel('Value', fontsize=20)
# plt.ylim([0.75, 0.87])
#
# cell = ['0', '1']
# width = 1
# index = np.arange(len(cell))
# figsize = (9, 6)  # 调整绘制图片的比例
# plt.bar('0', 0.8117, width, color="#87CEFA",)  # 绘制柱状图
# plt.bar('1', 0.8552, width=width, facecolor='yellowgreen')
# plt.ylabel('Value', fontsize=20)
# plt.xticks(index, cell, fontsize=20)  # 将横坐标用cell替换,fontsize用来调整字体的大小
# plt.text(0, 0.821, '0.8117', ha='center', va='top', fontsize=20)
# plt.text(1, 0.865, '0.8552', ha='center', va='top', fontsize=20)
# plt.show()



# 0.8276648101
# 0.9775511776488283
# 1
# 1.0188
# 1.0703091513
# for i in range(10):
#     b += '+' + str(para[i]) + '*(0.5)**' + str(i)
#
# # print(eval(b))
# t = np.linspace(-0.495, 4.495, num=5000)
# plt.figure(figsize=(9, 6))
# interval0 = [0.8276 if (i>=-0.5 and i<0.5) else 0 for i in t]
# interval1 = [0.9775 if (i>=0.5 and i<1.5) else 0 for i in t]
# interval2 = [1 if (i>=1.5 and i<2.5) else 0 for i in t]
# interval3 = [1.0188 if (i>=2.5 and i<3.5) else 0 for i in t]
# interval4 = [1.0703 if (i>=3.5 and i<4.5) else 0 for i in t]
# y = t/t*interval0 + interval1 + interval2 + interval3 + interval4
# plt.plot(t,y,'r-')
# plt.legend(['Work Type'], fontsize=20)
# cell = ['Children', 'Never work', 'Govt job', 'Self employed', 'Private']
# width = 1
# index = np.arange(len(cell))
# plt.bar('0', 0.8276, width, color="#87CEFA")  # 绘制柱状图
# plt.bar('1', 0.9775, width, facecolor='turquoise')
# plt.bar('2', 1, width, color="aqua",)  # 绘制柱状图
# plt.bar('3', 1.0188, width, color='palegreen')
# plt.bar('4', 1.0703, width, color="yellowgreen",)
# plt.ylim([0.8, 1.1])
# plt.ylabel('Value', fontsize=20)
# plt.xticks(index, cell, fontsize=15)  # 将横坐标用cell替换,fontsize用来调整字体的大小
# plt.text(0, 0.84, '0.8276', ha='center', va='top', fontsize=15)
# plt.text(1, 0.99, '0.9775', ha='center', va='top', fontsize=15)
# plt.text(2, 1.013, '1', ha='center', va='top', fontsize=15)
# plt.text(3, 1.031, '1.0188', ha='center', va='top', fontsize=15)
# plt.text(4, 1.0833, '1.0703', ha='center', va='top', fontsize=15)
# plt.show()



# # 0.9042883092
# # 0.8555466728
# for i in range(10):
#     b += '+' + str(para[i]) + '*(2*(81-21)/60-1)**' + str(i)
#
# # print(eval(b))
# t = np.linspace(-0.495, 1.495, num=500)
#
# y = [0.9042 if (i < 0.5) else 0.8555 for i in t]
# plt.figure(figsize=(9, 6))
# plt.yticks(fontsize=20, color='#000000')
# plt.xticks(fontsize=20, color='#000000')
# plt.plot(t, y, 'r-')
# plt.legend(['Residence Type'], fontsize=20)
# plt.ylabel('Value', fontsize=20)
# plt.ylim([0.8, 0.93])
#
# cell = ['Rural', 'Urban']
# width = 1
# index = np.arange(len(cell))
# figsize = (9, 6)  # 调整绘制图片的比例
# plt.bar('0', 0.9042, width, color="#87CEFA",)  # 绘制柱状图
# plt.bar('1', 0.8555, width=width, facecolor='yellowgreen')
# plt.ylabel('Value', fontsize=20)
# plt.xticks(index, cell, fontsize=20)  # 将横坐标用cell替换,fontsize用来调整字体的大小
# plt.text(0, 0.912, '0.9042', ha='center', va='top', fontsize=20)
# plt.text(1, 0.864, '0.8555', ha='center', va='top', fontsize=20)
# plt.show()




# for i in range(10):
#     b += '+' + str(para[i]) + '*(2*(t-55.1199)/216.6199-1)**' + str(i)
# print(b)
# t = np.linspace(55.12, 271.74, num=5000)
# y1 = f1()
# y2 = f2(t)
# plt.figure(figsize=(9, 6))
# plt.yticks(fontsize=20,color='#000000')
# plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
# plt.plot(t,y1,'r-')
# plt.plot(t,y2,'k--')
# plt.legend(['Avg Glucose Level'], fontsize=20)#图例
# # plt.xlabel('Age', fontsize=20)#设置x,y轴标记
# plt.ylabel('Value', fontsize=20)
# plt.ylim([0, 2.5])
# plt.show()



# for i in range(10):
#     b += '+' + str(para[i]) + '*(2*(t-11.5)/(92-11.5)-1)**' + str(i)
# print(b)
# t = np.linspace(11.5, 92, num=5000)
# y1 = f1()
# y2 = f2(t)
# plt.figure(figsize=(9, 6))
# plt.yticks(fontsize=20,color='#000000')
# plt.xticks(fontsize=20,color='#000000')  #不显示x轴刻度值
# plt.plot(t,y1,'r-')
# plt.plot(t,y2,'k--')
# plt.legend(['BMI'], fontsize=20)#图例
# # plt.xlabel('Age', fontsize=20)#设置x,y轴标记
# plt.ylabel('Value', fontsize=20)
# plt.ylim([0.8, 1.1])
# plt.show()




# 1.0129616952
# 1
# 1.0681525157
# for i in range(10):
#     b += '+' + str(para[i]) + '*(1)**' + str(i)
#
# # print(eval(b))
# t = np.linspace(-0.495, 2.495, num=5000)
# plt.figure(figsize=(9, 6))
# interval0 = [1.0129 if (i>=-0.5 and i<0.5) else 0 for i in t]
# interval1 = [1 if (i>=0.5 and i<1.5) else 0 for i in t]
# interval2 = [1.0681 if (i>=1.5 and i<2.5) else 0 for i in t]
#
# y = t/t*interval0 + interval1 + interval2
# plt.plot(t,y,'r-')
# plt.legend(['Smoking Status'], fontsize=20)
# cell = ['Never', 'Formerly Smoked', 'Smokes']
# width = 1
# index = np.arange(len(cell))
# plt.bar('0', 1.0129, width, color="#87CEFA")  # 绘制柱状图
# plt.bar('1', 1, width, facecolor='aqua')
# plt.bar('2', 1.0681, width, color="yellowgreen",)  # 绘制柱状图
# plt.ylim([0.95, 1.08])
# plt.ylabel('Value', fontsize=20)
# plt.xticks(index, cell, fontsize=15)  # 将横坐标用cell替换,fontsize用来调整字体的大小
# plt.text(0, 1.021, '1.0129', ha='center', va='top', fontsize=20)
# plt.text(1, 1.009, '1', ha='center', va='top', fontsize=20)
# plt.text(2, 1.077, '1.0681', ha='center', va='top', fontsize=20)
# plt.show()


