#%%程序说明:
'''
根据参考文献提供的方法,利用Davenport谱,
1.模拟生成脉动风速时程,脉动风压时程
2.绘制相关图像(风速时程,风压时程,模拟功率谱与理论功率谱)
3.写出生成的时程文件供导入midas分析使用
参考文献:
简谐波叠加法模拟风谱_阎石
结构随机振动_欧进萍
'''
#%%导包
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
# %%设置字体为TimesNewman
config = {
    "font.family":'serif',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
chineseFont='STSONG'
rcParams.update(config)

del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()
#%%导包
import numpy as np
import matplotlib.mlab as mlab
from matplotlib.pyplot import MultipleLocator, figure
from matplotlib import ticker

import xlsxwriter
#%%调用函数生成风速时程序列
K = 0.00215
p = 1.225  # 空气密度，单位kg/m3
N = 10240
w = np.linspace(0, 20, N).reshape(-1, 1)[1:, :]
v10_avg = 22
alpha = 0.16  # 风速沿高度变化
z = 170  # 目标高度
vz = v10_avg*(z/10)**alpha
print('vz', vz)
dt = 0.1
t = np.arange(0, 200, dt).reshape(-1, 1)
print('w的shape', w.shape)
print('t的shape', t.shape)
#%%定义davenport谱函数
np.random.seed(20) # 让随机序列不再变化
fai = 2*np.pi*np.random.random(w.shape).reshape(-1, 1)  # 生成随机均匀分布的相位角
dw = w[1]-w[0]  # 求圆频率间距
f = w/(2*np.pi)  # 换算成频率
x = 600*w/(v10_avg*np.pi)
# x = 1200*f/v10_avg # x的另一种表示
Sv = 4*K*(v10_avg**2)/w*x**2/((1+x**2)**(4/3))
# 公式参考:结构随机振动_欧进萍
Sw = p**2*vz**2*Sv
# 公式参考:简谐波叠加法模拟风谱_阎石
#%%算风速时程
#  g=np.sqrt(2*Sv*dw)*np.cos(w*t+fai)#把此公式分解为两部分计算，方便阅读
cal_1 = np.sqrt(2*Sv*dw)@np.ones_like(t.T)
cal_2 = np.cos(w@t.T+fai@np.ones_like(t.T))
print('cal_1', cal_1.shape)
print('cal_2', cal_2.shape)
avgcosfai = np.cos(fai).sum(axis=0)/w.size
print('avg_cos_fai', avgcosfai)
v_timehistory = (cal_1*cal_2).sum(axis=0).reshape(-1, 1)
print('v_timehistory的shape', v_timehistory.shape)
#%%算风压时程
#  g=np.sqrt(2*Sv*dw)*np.cos(w*t+fai)#把此公式分解为两部分计算，方便阅读
cal_3 = np.sqrt(2*Sw*dw)@np.ones_like(t.T)
cal_4 = np.cos(w@t.T+fai@np.ones_like(t.T))
print('cal_3', cal_1.shape)
print('cal_4', cal_2.shape)
avgcosfai = np.cos(fai).sum(axis=0)/w.size
print('avg_cos_fai', avgcosfai)
w_timehistory = (cal_3*cal_4).sum(axis=0).reshape(-1, 1)  # 单位：N/m2
print('w_timehistory_shape', w_timehistory.shape)
avgW = w_timehistory.sum(axis=0)/w_timehistory.size
maxW = max(w_timehistory)
print('avgW/maxW', avgW/maxW)
#%%绘图:风速时程的功率谱
# 设置绘图颜色
color1='r'
color2='g'
# 绘制图线
fs = 1/dt
v_timehistory1D = v_timehistory.flatten()
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.figure(2, figsize=(10, 5), dpi=60)
v_pxx_plot = plt.psd(v_timehistory1D, 1024*3, fs,
                     return_line=True, label='normal ordinate')
plt.figure(3, figsize=(5, 3), dpi=120)
plt1 = plt.loglog(v_pxx_plot[1], v_pxx_plot[0] /
                  fs*2, linewidth=1, label='log ordinate',color=color1)
l2, = plt.loglog(w/(2*np.pi), Sv, linewidth=1, linestyle='-.',color=color2)

# 标签字体设置
fontsizeAll = 12
plt.xlabel(r'$f/Hz$(频率)', fontsize=fontsizeAll,family=chineseFont)
plt.ylabel(r'$S_{xx}(f)/($'+' '+'$ {m}^{2}\cdot{s}^{-1})$',
           fontsize=fontsizeAll)
plt.xticks(fontsize=fontsizeAll)
plt.yticks(fontsize=fontsizeAll)
axes = plt.gca()
axes.set_xlim([1e-3, 1])
axes.set_ylim([1e-6, 1e2])
axes.yaxis.set_major_locator(ticker.LogLocator(base=100.0))
axes.yaxis.set_minor_locator(ticker.NullLocator())
axes.xaxis.set_minor_locator(ticker.NullLocator())
plt.legend(['模拟功率谱', r'$Davenport$功率谱'], fontsize=fontsizeAll,prop={"family" : chineseFont,"size":fontsizeAll},loc="lower left",frameon=False)
# plt.legend(['Simulated Spectrum', 'Davenport Spectrum'], fontsize=fontsizeAll)
plt.tight_layout()
plt.savefig("模拟脉动风速时程对应功率谱与Davenport功率谱.png")
#%%绘图:其他图
plt.figure(1, figsize=(5, 3), dpi=120)
l1, = plt.plot(t, v_timehistory, linewidth=1,color=color1)
# plt.title('脉动风速时程',fontproperties = 'SimHei',fontsize = 20)
fontsizeAll = 12
plt.ylabel(r'$v/(m/{s})$', fontsize=fontsizeAll)
plt.xlabel(r'$t/s$(时间)', fontsize=fontsizeAll,family=chineseFont)
plt.xticks(fontsize=fontsizeAll)
plt.yticks(fontsize=fontsizeAll)
axes = plt.gca()
axes.set_xlim([0, 200])
axes.set_ylim([-10, 10])
axes.yaxis.set_major_locator(MultipleLocator(4))
plt.tight_layout()
plt.savefig("模拟脉动风速时程.png")

plt.figure(4, figsize=(5, 3), dpi=120)
l3, = plt.plot(t, w_timehistory/1000, linewidth=1,color=color1)
# plt.title('脉动风压时程', fontproperties='SimHei', fontsize=20)
fontsizeAll = 12
plt.ylabel(r'$w/(kN/{m}^2)$', fontsize=fontsizeAll)
plt.xlabel(r'$t/s$(时间)', fontsize=fontsizeAll,family=chineseFont)
plt.xticks(fontsize=fontsizeAll)
plt.yticks(fontsize=fontsizeAll)
axes = plt.gca()
axes.set_xlim([0, 200])
# axes.set_ylim([-10, 10])
# axes.yaxis.set_major_locator(MultipleLocator(4))
plt.tight_layout()
plt.savefig("模拟脉动风压时程.png")
#%%保存为excel:
#定义写入Excel的函数
def writeVectorToXlsx(vector,worksheet,column,colName):
    idx=1
    worksheet.write(0,column,colName)
    for i in vector:
        worksheet.write(idx,column,i)
        idx+=1
    print(colName+"写入成功")
    return

excelData=xlsxwriter.Workbook('wind_simulation.xlsx')
t1= excelData.add_worksheet('velocity_timehistory')
t21= excelData.add_worksheet('Spectrum-Davenport谱')
t22= excelData.add_worksheet('Spectrum-模拟谱')
t3= excelData.add_worksheet('pressure_timehistory')
# 写出t1-风速时程
writeVectorToXlsx(t,t1,0,"t/s")
writeVectorToXlsx(v_timehistory,t1,1,"v_timehistory/(m/s)")
# 写出t2-风速功率谱
writeVectorToXlsx(w/(2*np.pi),t21,0,"f/Hz")
writeVectorToXlsx(Sv,t21,1,"Davenport谱-Sxx(f)/(m2*s-1")
t21.write(0,6,"本表数据为由模拟风速时程得到的功率谱,请用双对数坐标绘制")
writeVectorToXlsx(v_pxx_plot[1],t22,0,"f/Hz")
writeVectorToXlsx(v_pxx_plot[0]/fs*2,t22,1,"模拟谱-Sxx(f)/(m2*s-1")
t22.write(0,6,"本表数据为由Davenport谱得到的功率谱,请用双对数坐标绘制")
# 写出t3-风压时程
writeVectorToXlsx(t,t3,0,"t/s")
writeVectorToXlsx(w_timehistory/1000,t3,1,"w_timehistory/(kN/m2)")
excelData.close()
#%%显示图片
plt.show()
#%%写出生成的时程文件
#写出的文件为midas用户自定义时程文件，后缀.thd,可导入midas进行时程分析
#格式如下：
# *UNIT, M , N - Length: MM, CM, M, INCH, FEET, GRAV allowed Load: KG, TON, KN, LBF, KIP
# *TYPE, ACCEL - ACCEL, FORCE, MOMENT allowed
# *Data
# X1, Y1 (x: Time, X: Time Function)
# X2, Y2
# X3, Y3
filename = 'wind timehistory.thd'
with open(filename, 'w') as file_object:
    file_object.write("*UNIT,M,KN\n")
    file_object.write("*TYPE,FORCE\n")
    file_object.write("*Data\n")
    # 为了在软件中顺利计算，强制使荷载起始值为0，并舍弃时程最后一个数据
    file_object.write("0.00000000,0.00000000\n")
    for i in range(len(w_timehistory)-1):
        t_str = "{:.8f}".format(t[i+1, 0])
        w_str = "{:.8f}".format(w_timehistory[i, 0])
        file_object.write(t_str)
        file_object.write(",")
        file_object.write(w_str)
        file_object.write("\n")
        # file_object.write(t(i,1),',',w_timehistory(i,1),'/n')
