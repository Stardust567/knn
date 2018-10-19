import numpy as np
import matplotlib.pyplot as plt

def fileToMatrix(filename):
    with open(filename) as fr:
        tempLines = fr.readlines()
        numberOfLines = len(tempLines)
        attrMat = np.zeros((numberOfLines, 3))
        index = 0
        labelMat = []
        for line in tempLines:
            line = line.strip().split('\t')
            attrMat[index,:] = line[0:len(line)-1];index += 1
            label = line[-1]
            if (label == "largeDoses"):
                label = 0
            elif (label == "smallDoses"):
                label = 1
            else:
                label = 2
            labelMat.append(label)
    return attrMat,labelMat

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, minVals, ranges

def getPicture(attrMat, labelMat):
    x = attrMat[:,0]
    y = attrMat[:,1]
    plt.figure("classify")
    plt.title("classify")
    plt.xlabel("Frequent flyer mileage(each year)")
    plt.ylabel("Percentage of time spent playing video games")
    plt.scatter(x,y,s=20*np.array(labelMat),c=np.array(labelMat),marker='*')
    plt.show()

def classifyOfDistance(attr, dataSet, labels, k):
    attr = np.tile(attr, (dataSet.shape[0],1)) - dataSet
    attr = np.sqrt(attr.sum(axis=1))
    sort_index = attr.argsort()
    labelsCount = np.zeros((dataSet.shape[1],1))
    for i in range(k):
        label = labels[sort_index[i]]
        labelsCount[label] += 1
    return (labelsCount.argmax())

def classifyOfCos(attr, dataSet, labels, k):
    dataCos = []
    for data in dataSet :
        attr = np.array(attr)
        sum_vector = (attr * data)
        sum_vector = np.array(sum_vector).sum()
        sum_attr = (attr **2).sum()
        sum_data = (data **2).sum()
        sum_num = np.sqrt(sum_attr * sum_data)
        cos = sum_vector / sum_num
        dataCos.append(cos)
    sort_index = np.array(dataCos).argsort()
    labelsCount = np.zeros((dataSet.shape[1], 1))
    for i in range(k):
        label = labels[sort_index[i]]
        labelsCount[label] += 1
    return (labelsCount.argmax())

def test(filename, textname,k):
    error_distance = 0
    error_cos = 0
    datingDataMat, datingLabels = fileToMatrix(filename)
    datingDataMat, minVals, ranges = autoNorm(datingDataMat)
    testingDataMat, testingLabels = fileToMatrix(textname)
    test_index = 0
    for test in testingDataMat:
        test = (np.array(test) - minVals)/ranges
        label_distance = classifyOfDistance(test, datingDataMat, datingLabels, k)
        label_cos = classifyOfCos(test, datingDataMat, datingLabels, k)
        if (label_distance != datingLabels[test_index]):
            error_distance += 1
        if (label_cos != datingLabels[test_index]):
            error_cos += 1
        test_index += 1
    num = len(testingDataMat)
    print(error_distance / num, error_cos / num)

if __name__ == '__main__':
    filename = 'datingTestSet.txt'
    textname = 'datingTestSet2.txt'
    datingDataMat, datingLabels = fileToMatrix(filename)
    testingDataMat, testingLabels = fileToMatrix(textname)
    datingDataMat, minVals, ranges = autoNorm(datingDataMat)
    testingDataMat, testingLabels = fileToMatrix(textname)
    test(filename, textname,1)
    #getPicture(datingDataMat,datingLabels)