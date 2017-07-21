import kNN
import drawgraph
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
'''
group,labels = kNN.createDataSet()
print (type(group))
print (type(labels))
print(kNN.classify0([0,0],group,labels,3))

datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
datingDataMat,range,minVal = kNN.autoNorm(datingDataMat)
print(type(datingDataMat))
print(type(datingLabels))
print(shape(datingDataMat))
print(len(datingLabels))
print(datingDataMat)
#drawgraph.drawgf(datingDataMat,datingLabels,1,0)

#kNN.datingClassTest()
kNN.classifyPerson()

kNN.handwritingClassTest()



print(type(fileMat))


mat = zeros((32,32,3))
height = int(mat.shape[0])
width =int(mat.shape[1])
for x in range(100):
    for i in range(height):
        for j in range(width):
            for k in range(3):
                mat[i][j][k]=imgDataMat[x][1024*k+height*i+j]

    plt.subplot(10,10,x+1)
    plt.imshow(mat,cmap=None)
plt.show()
'''
kNN.CIFAR_10ClassTrest()

