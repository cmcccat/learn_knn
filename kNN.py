from numpy import *
from os import listdir
import random
def createDataSet():
    #numpy array
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#kNN算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #tile:numpy函数，将inX重复dataSetSize行列，便于进行计算
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #**表示乘方运算
    sqDiffMat = diffMat ** 2
    #求和 axis = 0列相加，axis =1 行相加
    sqDistances = sqDiffMat.sum(axis =1 )
    distances = sqDistances ** 0.5
    #返回数组从小到大下标的索引
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #返回前k个值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #dict.get(key,default=None)获取下标值，不存在就设置其默认值（对应第二个参数）
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #多列排序，第二参数为所选的列，第三个参数表示是否降序排列
    sortedClassCount = sorted(classCount.items(), key = lambda item:item[1],reverse = True)
    #获取第一行第一列也就是最大值的下标
    return sortedClassCount[0][0]

#将文本数据转换为numpy矩阵
def file2matrix(filename):
     fr = open(filename)
     arrayOfLines = fr.readlines()
     numberOfLines = len(arrayOfLines)
     #构造初始化矩阵
     returnMat = zeros((numberOfLines,3))
     classLabelVector = []
     index = 0
     for line in arrayOfLines:
         #清除两边空格之类的字符
         line = line.strip()
         #以空格为分割
         listFormLine = line.split('\t')
         returnMat[index,:] = listFormLine[0:3]
         #返回的最后一列作为标签
         classLabelVector.append(int(listFormLine[-1]))
         index += 1
     return returnMat,classLabelVector

#数据归一化
def autoNorm(dataSet):
    #0:列最小，1:行最小
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    range = maxVals-minVals
    #zeros 属于array
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(range,(m,1))
    return normDataSet,range,minVals

#约会网站数据测试
def datingClassTest():
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,rangeX,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:{},the real answer is:{}".format(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount += 1.0
    print("the total error rate is:{}".format(errorCount/float(numTestVecs)))

#分类算法（交互）
def classifyPerson():
    resultList =['完全没有魅力','有点兴趣','非常有魅力']
    percentTats = float(input("玩视频游戏百分比？"))
    ffMiles = float(input("每年飞行里数？"))
    iceCream = float(input("每年吃冰激凌公升数？"))
    datingDataMat,datingLabels =file2matrix('datingTestSet2.txt')
    normMat, rangeX, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/rangeX,normMat,datingLabels,3)
    print("你可能会喜欢这样的人：",resultList[classifierResult-1])

#将手写数字转换为向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#手写数字识别测试代码
def handwritingClassTest():
    hwLabels  = []
    #listdir 就是获取某一文件夹下文件名列表
    trainingFileList = listdir('trainingDigits')
    print(trainingFileList)
    m = len(trainingFileList)
    #我们的测试数据组，将图片信息拓展成一个1024维的向量
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("trainingDigits/{}".format(fileNameStr))
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:{},the real answer is:{}".format(classifierResult,classNumStr))
        if (classifierResult != classNumStr):errorCount += 1.0
    print("the total number of errors is；{}".format(errorCount))
    print("the tota; error rate is:{}".format(errorCount/float(mTest)))

#读取CIFAR-10数据集
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

#将CIFAR-10解析为向量
def img2vector1():
    imgDataMat = []
    labelsMat = []
    for i in range(5):
        batch1 = unpickle("cifar-10-batches-py/data_batch_{}".format(i+1))
        for key in batch1:
            if (key.decode() == 'data'):
                imgDataMat.extend(batch1[key])
            if (key.decode() == 'labels'):
                labelsMat.extend(batch1[key])
    imgDataMat = array(imgDataMat)
    return imgDataMat,labelsMat

#CIFAR-10 KNN识别
def CIFAR_10ClassTrest():
    trainimgMat,trainlabelsMat = img2vector1()
    batch1 = unpickle("cifar-10-batches-py/test_batch")
    for key in batch1:
        if (key.decode() == 'data'):
            testimgMat = array(batch1[key])
        if (key.decode() == 'labels'):
            testlabelsMat = batch1[key]
    mTest = len(testlabelsMat)
    print(mTest)
    errorCount = 0.0
    for i in range(500):
        randindex = random.randint(0,mTest-1)
        classifierResult = classify0(testimgMat[randindex], trainimgMat, trainlabelsMat, 7)
        print("the index is {},the classifier came back with:{},the real answer is:{}".format(randindex,classifierResult, testlabelsMat[randindex]))
        if (classifierResult != testlabelsMat[randindex]): errorCount += 1.0
    print("the total number of errors is；{}".format(errorCount))
    print("the tota; error rate is:{}".format(errorCount / float(400)))
