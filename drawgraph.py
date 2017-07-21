import matplotlib
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
from numpy import *
#画图
def drawgf(datingDataMat,datingLabels,x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 将三类数据分别取出来
    # x轴代表飞行的里程数
    # y轴代表玩视频游戏的百分比
    s0 = u'每年获取的飞行里程数'
    s1 = u'玩视频游戏所消耗的事件百分比'
    s2 = u'每周所消费冰激凌公升数'
    message = {0:s0,1:s1,2:s2}
    sx = message[x]
    sy = message[y]
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:  # 不喜欢
            type1_x.append(datingDataMat[i][int(x)])
            type1_y.append(datingDataMat[i][int(y)])

        if datingLabels[i] == 2:  # 魅力一般
            type2_x.append(datingDataMat[i][int(x)])
            type2_y.append(datingDataMat[i][int(y)])

        if datingLabels[i] == 3:  # 极具魅力
            type3_x.append(datingDataMat[i][int(x)])
            type3_y.append(datingDataMat[i][int(y)])
    type1 = ax.scatter(type1_x, type1_y, s=20, c='red')
    type2 = ax.scatter(type2_x, type2_y, s=40, c='green')
    type3 = ax.scatter(type3_x, type3_y, s=50, c='blue')
    plt.xlabel(sx)
    plt.ylabel(sy)
    plt.title('约会男士魅力可视化')
    ax.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)
    plt.show()