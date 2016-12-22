from sklearn.svm import SVC
import cv2
import numpy as np
import math

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

listing1 = []
data_train1 =[]
data_train2 =[]
data_train3 =[]
data_train4 =[]
data_train5 =[]

data_test1 = []
data_test2 = []
data_test3 = []
data_test4 = []
data_test5 = []

for i in range(1,201):
    listing1.append("c"+format(i)+ ".png")
    if i <41:
        data_train1.append("c" + format(i) + ".png")
        data_train2.append("c" + format(i) + ".png")
        data_train3.append("c" + format(i) + ".png")
        data_train4.append("c" + format(i) + ".png")
        data_test5.append("c" + format(i) + ".png")
    elif i<81:
        data_train1.append("c" + format(i) + ".png")
        data_train4.append("c" + format(i) + ".png")
        data_train3.append("c" + format(i) + ".png")
        data_train5.append("c" + format(i) + ".png")
        data_test2.append("c" + format(i) + ".png")
    elif i <121:
        data_train1.append("c" + format(i) + ".png")
        data_train2.append("c" + format(i) + ".png")
        data_train4.append("c" + format(i) + ".png")
        data_train5.append("c" + format(i) + ".png")
        data_test3.append("c" + format(i) + ".png")
    elif i<161:
        data_train4.append("c" + format(i) + ".png")
        data_train2.append("c" + format(i) + ".png")
        data_train3.append("c" + format(i) + ".png")
        data_train5.append("c" + format(i) + ".png")
        data_test1.append("c" + format(i) + ".png")
    else:
        data_train1.append("c" + format(i) + ".png")
        data_train2.append("c" + format(i) + ".png")
        data_train3.append("c" + format(i) + ".png")
        data_train5.append("c" + format(i) + ".png")
        data_test4.append("c" + format(i) + ".png")
for i in range(1,201):
    listing1.append("d"+format(i)+ ".png")
    if i < 41:
        data_train1.append("d" + format(i) + ".png")
        data_train2.append("d" + format(i) + ".png")
        data_train3.append("d" + format(i) + ".png")
        data_train4.append("d" + format(i) + ".png")
        data_test5.append("d" + format(i) + ".png")
    elif i < 81:
        data_train1.append("d" + format(i) + ".png")
        data_train4.append("d" + format(i) + ".png")
        data_train3.append("d" + format(i) + ".png")
        data_train5.append("d" + format(i) + ".png")
        data_test2.append("d" + format(i) + ".png")
    elif i < 121:
        data_train1.append("d" + format(i) + ".png")
        data_train2.append("d" + format(i) + ".png")
        data_train4.append("d" + format(i) + ".png")
        data_train5.append("d" + format(i) + ".png")
        data_test3.append("d" + format(i) + ".png")
    elif i < 161:
        data_train4.append("d" + format(i) + ".png")
        data_train2.append("d" + format(i) + ".png")
        data_train3.append("d" + format(i) + ".png")
        data_train5.append("d" + format(i) + ".png")
        data_test1.append("d" + format(i) + ".png")
    else:
        data_train1.append("d" + format(i) + ".png")
        data_train2.append("d" + format(i) + ".png")
        data_train3.append("d" + format(i) + ".png")
        data_train5.append("d" + format(i) + ".png")
        data_test4.append("d" + format(i) + ".png")
for i in range(1,201):
    listing1.append("f"+format(i)+ ".png")
    if i < 41:
        data_train1.append("f" + format(i) + ".png")
        data_train2.append("f" + format(i) + ".png")
        data_train3.append("f" + format(i) + ".png")
        data_train4.append("f" + format(i) + ".png")
        data_test5.append("f" + format(i) + ".png")
    elif i < 81:
        data_train1.append("f" + format(i) + ".png")
        data_train4.append("f" + format(i) + ".png")
        data_train3.append("f" + format(i) + ".png")
        data_train5.append("f" + format(i) + ".png")
        data_test2.append("f" + format(i) + ".png")
    elif i < 121:
        data_train1.append("f" + format(i) + ".png")
        data_train2.append("f" + format(i) + ".png")
        data_train4.append("f" + format(i) + ".png")
        data_train5.append("f" + format(i) + ".png")
        data_test3.append("f" + format(i) + ".png")
    elif i < 161:
        data_train4.append("f" + format(i) + ".png")
        data_train2.append("f" + format(i) + ".png")
        data_train3.append("f" + format(i) + ".png")
        data_train5.append("f" + format(i) + ".png")
        data_test1.append("f" + format(i) + ".png")
    else:
        data_train1.append("f" + format(i) + ".png")
        data_train2.append("f" + format(i) + ".png")
        data_train3.append("f" + format(i) + ".png")
        data_train5.append("f" + format(i) + ".png")
        data_test4.append("f" + format(i) + ".png")
for i in range(1,201):
    listing1.append("five"+format(i)+ ".png")
    if i < 41:
        data_train1.append("five" + format(i) + ".png")
        data_train2.append("five" + format(i) + ".png")
        data_train3.append("five" + format(i) + ".png")
        data_train4.append("five" + format(i) + ".png")
        data_test5.append("five" + format(i) + ".png")
    elif i < 81:
        data_train1.append("five" + format(i) + ".png")
        data_train4.append("five" + format(i) + ".png")
        data_train3.append("five" + format(i) + ".png")
        data_train5.append("five" + format(i) + ".png")
        data_test2.append("five" + format(i) + ".png")
    elif i < 121:
        data_train1.append("five" + format(i) + ".png")
        data_train2.append("five" + format(i) + ".png")
        data_train4.append("five" + format(i) + ".png")
        data_train5.append("five" + format(i) + ".png")
        data_test3.append("five" + format(i) + ".png")
    elif i < 161:
        data_train4.append("five" + format(i) + ".png")
        data_train2.append("five" + format(i) + ".png")
        data_train3.append("five" + format(i) + ".png")
        data_train5.append("five" + format(i) + ".png")
        data_test1.append("five" + format(i) + ".png")
    else:
        data_train1.append("five" + format(i) + ".png")
        data_train2.append("five" + format(i) + ".png")
        data_train3.append("five" + format(i) + ".png")
        data_train5.append("five" + format(i) + ".png")
        data_test4.append("five" + format(i) + ".png")
for i in range(1,201):
    listing1.append("four"+format(i)+ ".png")
    if i < 41:
        data_train1.append("four" + format(i) + ".png")
        data_train2.append("four" + format(i) + ".png")
        data_train3.append("four" + format(i) + ".png")
        data_train4.append("four" + format(i) + ".png")
        data_test5.append("four" + format(i) + ".png")
    elif i < 81:
        data_train1.append("four" + format(i) + ".png")
        data_train4.append("four" + format(i) + ".png")
        data_train3.append("four" + format(i) + ".png")
        data_train5.append("four" + format(i) + ".png")
        data_test2.append("four" + format(i) + ".png")
    elif i < 121:
        data_train1.append("four" + format(i) + ".png")
        data_train2.append("four" + format(i) + ".png")
        data_train4.append("four" + format(i) + ".png")
        data_train5.append("four" + format(i) + ".png")
        data_test3.append("four" + format(i) + ".png")
    elif i < 161:
        data_train4.append("four" + format(i) + ".png")
        data_train2.append("four" + format(i) + ".png")
        data_train3.append("four" + format(i) + ".png")
        data_train5.append("four" + format(i) + ".png")
        data_test1.append("four" + format(i) + ".png")
    else:
        data_train1.append("four" + format(i) + ".png")
        data_train2.append("four" + format(i) + ".png")
        data_train3.append("four" + format(i) + ".png")
        data_train5.append("four" + format(i) + ".png")
        data_test4.append("four" + format(i) + ".png")
for i in range(1,201):
    listing1.append("g"+format(i)+".png")
    if i < 41:
        data_train1.append("g" + format(i) + ".png")
        data_train2.append("g" + format(i) + ".png")
        data_train3.append("g" + format(i) + ".png")
        data_train4.append("g" + format(i) + ".png")
        data_test5.append("g" + format(i) + ".png")
    elif i < 81:
        data_train1.append("g" + format(i) + ".png")
        data_train4.append("g" + format(i) + ".png")
        data_train3.append("g" + format(i) + ".png")
        data_train5.append("g" + format(i) + ".png")
        data_test2.append("g" + format(i) + ".png")
    elif i < 121:
        data_train1.append("g" + format(i) + ".png")
        data_train2.append("g" + format(i) + ".png")
        data_train4.append("g" + format(i) + ".png")
        data_train5.append("g" + format(i) + ".png")
        data_test3.append("g" + format(i) + ".png")
    elif i < 161:
        data_train4.append("g" + format(i) + ".png")
        data_train2.append("g" + format(i) + ".png")
        data_train3.append("g" + format(i) + ".png")
        data_train5.append("g" + format(i) + ".png")
        data_test1.append("g" + format(i) + ".png")
    else:
        data_train1.append("g" + format(i) + ".png")
        data_train2.append("g" + format(i) + ".png")
        data_train3.append("g" + format(i) + ".png")
        data_train5.append("g" + format(i) + ".png")
        data_test4.append("g" + format(i) + ".png")
for i in range(1,201):
    listing1.append("i"+format(i)+".png")
    if i < 41:
        data_train1.append("i" + format(i) + ".png")
        data_train2.append("i" + format(i) + ".png")
        data_train3.append("i" + format(i) + ".png")
        data_train4.append("i" + format(i) + ".png")
        data_test5.append("i" + format(i) + ".png")
    elif i < 81:
        data_train1.append("i" + format(i) + ".png")
        data_train4.append("i" + format(i) + ".png")
        data_train3.append("i" + format(i) + ".png")
        data_train5.append("i" + format(i) + ".png")
        data_test2.append("i" + format(i) + ".png")
    elif i < 121:
        data_train1.append("i" + format(i) + ".png")
        data_train2.append("i" + format(i) + ".png")
        data_train4.append("i" + format(i) + ".png")
        data_train5.append("i" + format(i) + ".png")
        data_test3.append("i" + format(i) + ".png")
    elif i < 161:
        data_train4.append("i" + format(i) + ".png")
        data_train2.append("i" + format(i) + ".png")
        data_train3.append("i" + format(i) + ".png")
        data_train5.append("i" + format(i) + ".png")
        data_test1.append("i" + format(i) + ".png")
    else:
        data_train1.append("i" + format(i) + ".png")
        data_train2.append("i" + format(i) + ".png")
        data_train3.append("i" + format(i) + ".png")
        data_train5.append("i" + format(i) + ".png")
        data_test4.append("i" + format(i) + ".png")
for i in range(1,201):
    listing1.append("l"+format(i)+".png")
    if i < 41:
        data_train1.append("l" + format(i) + ".png")
        data_train2.append("l" + format(i) + ".png")
        data_train3.append("l" + format(i) + ".png")
        data_train4.append("l" + format(i) + ".png")
        data_test5.append("l" + format(i) + ".png")
    elif i < 81:
        data_train1.append("l" + format(i) + ".png")
        data_train4.append("l" + format(i) + ".png")
        data_train3.append("l" + format(i) + ".png")
        data_train5.append("l" + format(i) + ".png")
        data_test2.append("l" + format(i) + ".png")
    elif i < 121:
        data_train1.append("l" + format(i) + ".png")
        data_train2.append("l" + format(i) + ".png")
        data_train4.append("l" + format(i) + ".png")
        data_train5.append("l" + format(i) + ".png")
        data_test3.append("l" + format(i) + ".png")
    elif i < 161:
            data_train4.append("l" + format(i) + ".png")
            data_train2.append("l" + format(i) + ".png")
            data_train3.append("l" + format(i) + ".png")
            data_train5.append("l" + format(i) + ".png")
            data_test1.append("l" + format(i) + ".png")
    else:
        data_train1.append("l" + format(i) + ".png")
        data_train2.append("l" + format(i) + ".png")
        data_train3.append("l" + format(i) + ".png")
        data_train5.append("l" + format(i) + ".png")
        data_test4.append("l" + format(i) + ".png")
for i in range(1,201):
    listing1.append("nine"+format(i)+".png")
    if i < 41:
        data_train1.append("nine" + format(i) + ".png")
        data_train2.append("nine" + format(i) + ".png")
        data_train3.append("nine" + format(i) + ".png")
        data_train4.append("nine" + format(i) + ".png")
        data_test5.append("nine" + format(i) + ".png")
    elif i < 81:
        data_train1.append("nine" + format(i) + ".png")
        data_train4.append("nine" + format(i) + ".png")
        data_train3.append("nine" + format(i) + ".png")
        data_train5.append("nine" + format(i) + ".png")
        data_test2.append("nine" + format(i) + ".png")
    elif i < 121:
        data_train1.append("nine" + format(i) + ".png")
        data_train2.append("nine" + format(i) + ".png")
        data_train4.append("nine" + format(i) + ".png")
        data_train5.append("nine" + format(i) + ".png")
        data_test3.append("nine" + format(i) + ".png")
    elif i < 161:
        data_train4.append("nine" + format(i) + ".png")
        data_train2.append("nine" + format(i) + ".png")
        data_train3.append("nine" + format(i) + ".png")
        data_train5.append("nine" + format(i) + ".png")
        data_test1.append("nine" + format(i) + ".png")
    else:
        data_train1.append("nine" + format(i) + ".png")
        data_train2.append("nine" + format(i) + ".png")
        data_train3.append("nine" + format(i) + ".png")
        data_train5.append("nine" + format(i) + ".png")
        data_test4.append("nine" + format(i) + ".png")
for i in range(1,201):
    listing1.append("one"+format(i)+".png")
    if i < 41:
        data_train1.append("one" + format(i) + ".png")
        data_train2.append("one" + format(i) + ".png")
        data_train3.append("one" + format(i) + ".png")
        data_train4.append("one" + format(i) + ".png")
        data_test5.append("one" + format(i) + ".png")
    elif i < 81:
        data_train1.append("one" + format(i) + ".png")
        data_train4.append("one" + format(i) + ".png")
        data_train3.append("one" + format(i) + ".png")
        data_train5.append("one" + format(i) + ".png")
        data_test2.append("one" + format(i) + ".png")
    elif i < 121:
        data_train1.append("one" + format(i) + ".png")
        data_train2.append("one" + format(i) + ".png")
        data_train4.append("one" + format(i) + ".png")
        data_train5.append("one" + format(i) + ".png")
        data_test3.append("one" + format(i) + ".png")
    elif i < 161:
        data_train4.append("one" + format(i) + ".png")
        data_train2.append("one" + format(i) + ".png")
        data_train3.append("one" + format(i) + ".png")
        data_train5.append("one" + format(i) + ".png")
        data_test1.append("one" + format(i) + ".png")
    else:
        data_train1.append("one" + format(i) + ".png")
        data_train2.append("one" + format(i) + ".png")
        data_train3.append("one" + format(i) + ".png")
        data_train5.append("one" + format(i) + ".png")
        data_test4.append("one" + format(i) + ".png")
for i in range(1,201):
    listing1.append("r"+format(i)+".png")
    if i < 41:
        data_train1.append("r" + format(i) + ".png")
        data_train2.append("r" + format(i) + ".png")
        data_train3.append("r" + format(i) + ".png")
        data_train4.append("r" + format(i) + ".png")
        data_test5.append("r" + format(i) + ".png")
    elif i < 81:
        data_train1.append("r" + format(i) + ".png")
        data_train4.append("r" + format(i) + ".png")
        data_train3.append("r" + format(i) + ".png")
        data_train5.append("r" + format(i) + ".png")
        data_test2.append("r" + format(i) + ".png")
    elif i < 121:
        data_train1.append("r" + format(i) + ".png")
        data_train2.append("r" + format(i) + ".png")
        data_train4.append("r" + format(i) + ".png")
        data_train5.append("r" + format(i) + ".png")
        data_test3.append("r" + format(i) + ".png")
    elif i < 161:
        data_train4.append("r" + format(i) + ".png")
        data_train2.append("r" + format(i) + ".png")
        data_train3.append("r" + format(i) + ".png")
        data_train5.append("r" + format(i) + ".png")
        data_test1.append("r" + format(i) + ".png")
    else:
        data_train1.append("r" + format(i) + ".png")
        data_train2.append("r" + format(i) + ".png")
        data_train3.append("r" + format(i) + ".png")
        data_train5.append("r" + format(i) + ".png")
        data_test4.append("r" + format(i) + ".png")
for i in range(1,201):
    listing1.append("three"+format(i)+".png")
    if i < 41:
        data_train1.append("three" + format(i) + ".png")
        data_train2.append("three" + format(i) + ".png")
        data_train3.append("three" + format(i) + ".png")
        data_train4.append("three" + format(i) + ".png")
        data_test5.append("three" + format(i) + ".png")
    elif i < 81:
        data_train1.append("three" + format(i) + ".png")
        data_train4.append("three" + format(i) + ".png")
        data_train3.append("three" + format(i) + ".png")
        data_train5.append("three" + format(i) + ".png")
        data_test2.append("three" + format(i) + ".png")
    elif i < 121:
        data_train1.append("three" + format(i) + ".png")
        data_train2.append("three" + format(i) + ".png")
        data_train4.append("three" + format(i) + ".png")
        data_train5.append("three" + format(i) + ".png")
        data_test3.append("three" + format(i) + ".png")
    elif i < 161:
        data_train4.append("three" + format(i) + ".png")
        data_train2.append("three" + format(i) + ".png")
        data_train3.append("three" + format(i) + ".png")
        data_train5.append("three" + format(i) + ".png")
        data_test1.append("three" + format(i) + ".png")
    else:
        data_train1.append("three" + format(i) + ".png")
        data_train2.append("three" + format(i) + ".png")
        data_train3.append("three" + format(i) + ".png")
        data_train5.append("three" + format(i) + ".png")
        data_test4.append("three" + format(i) + ".png")
for i in range(1,201):
    listing1.append("two"+format(i)+ ".png")
    if i < 41:
        data_train1.append("two" + format(i) + ".png")
        data_train2.append("two" + format(i) + ".png")
        data_train3.append("two" + format(i) + ".png")
        data_train4.append("two" + format(i) + ".png")
        data_test5.append("two" + format(i) + ".png")
    elif i < 81:
        data_train1.append("two" + format(i) + ".png")
        data_train4.append("two" + format(i) + ".png")
        data_train3.append("two" + format(i) + ".png")
        data_train5.append("two" + format(i) + ".png")
        data_test2.append("two" + format(i) + ".png")
    elif i < 121:
        data_train1.append("two" + format(i) + ".png")
        data_train2.append("two" + format(i) + ".png")
        data_train4.append("two" + format(i) + ".png")
        data_train5.append("two" + format(i) + ".png")
        data_test3.append("two" + format(i) + ".png")
    elif i < 161:
        data_train4.append("two" + format(i) + ".png")
        data_train2.append("two" + format(i) + ".png")
        data_train3.append("two" + format(i) + ".png")
        data_train5.append("two" + format(i) + ".png")
        data_test1.append("two" + format(i) + ".png")
    else:
        data_train1.append("two" + format(i) + ".png")
        data_train2.append("two" + format(i) + ".png")
        data_train3.append("two" + format(i) + ".png")
        data_train5.append("two" + format(i) + ".png")
        data_test4.append("two" + format(i) + ".png")
for i in range(1,201):
    listing1.append("u"+format(i)+ ".png")
    if i < 41:
        data_train1.append("u" + format(i) + ".png")
        data_train2.append("u" + format(i) + ".png")
        data_train3.append("u" + format(i) + ".png")
        data_train4.append("u" + format(i) + ".png")
        data_test5.append("u" + format(i) + ".png")
    elif i < 81:
        data_train1.append("u" + format(i) + ".png")
        data_train4.append("u" + format(i) + ".png")
        data_train3.append("u" + format(i) + ".png")
        data_train5.append("u" + format(i) + ".png")
        data_test2.append("u" + format(i) + ".png")
    elif i < 121:
        data_train1.append("u" + format(i) + ".png")
        data_train2.append("u" + format(i) + ".png")
        data_train4.append("u" + format(i) + ".png")
        data_train5.append("u" + format(i) + ".png")
        data_test3.append("u" + format(i) + ".png")
    elif i < 161:
        data_train4.append("u" + format(i) + ".png")
        data_train2.append("u" + format(i) + ".png")
        data_train3.append("u" + format(i) + ".png")
        data_train5.append("u" + format(i) + ".png")
        data_test1.append("u" + format(i) + ".png")
    else:
        data_train1.append("u" + format(i) + ".png")
        data_train2.append("u" + format(i) + ".png")
        data_train3.append("u" + format(i) + ".png")
        data_train5.append("u" + format(i) + ".png")
        data_test4.append("u" + format(i) + ".png")
for i in range(1,201):
    listing1.append("v"+format(i)+".png")
    if i < 41:
        data_train1.append("v" + format(i) + ".png")
        data_train2.append("v" + format(i) + ".png")
        data_train3.append("v" + format(i) + ".png")
        data_train4.append("v" + format(i) + ".png")
        data_test5.append("v" + format(i) + ".png")
    elif i < 81:
        data_train1.append("v" + format(i) + ".png")
        data_train4.append("v" + format(i) + ".png")
        data_train3.append("v" + format(i) + ".png")
        data_train5.append("v" + format(i) + ".png")
        data_test2.append("v" + format(i) + ".png")
    elif i < 121:
        data_train1.append("v" + format(i) + ".png")
        data_train2.append("v" + format(i) + ".png")
        data_train4.append("v" + format(i) + ".png")
        data_train5.append("v" + format(i) + ".png")
        data_test3.append("v" + format(i) + ".png")
    elif i < 161:
        data_train4.append("v" + format(i) + ".png")
        data_train2.append("v" + format(i) + ".png")
        data_train3.append("v" + format(i) + ".png")
        data_train5.append("v" + format(i) + ".png")
        data_test1.append("v" + format(i) + ".png")
    else:
        data_train1.append("v" + format(i) + ".png")
        data_train2.append("v" + format(i) + ".png")
        data_train3.append("v" + format(i) + ".png")
        data_train5.append("v" + format(i) + ".png")
        data_test4.append("v" + format(i) + ".png")
#
# new_listing = []
# for i in range(3000):
#     new_listing.append(listing1[i]+".png")

training_set = []
testing_set = []
training_y = []
for i in range(2401):
    if i > 0 and i < 161:
        training_y.append(0)
    elif i > 160 and i < 321:
        training_y.append(1)
    elif i>320 and i<481:
        training_y.append(2)
    elif i>480 and i<641:
        training_y.append(3)
    elif i>640 and i<801:
        training_y.append(4)
    elif i>800 and i<961:
        training_y.append(5)
    elif i > 960 and i < 1121:
        training_y.append(6)
    elif i > 1120 and i < 1281:
        training_y.append(7)
    elif i > 1280 and i < 1441:
        training_y.append(8)
    elif i >1440 and i < 1601:
        training_y.append(9)
    elif i > 1600 and i < 1761:
        training_y.append(10)
    elif i > 1760 and i < 1921:
        training_y.append(11)
    elif i > 1920 and i < 2081:
        training_y.append(12)
    elif i > 2080 and i <2241:
        training_y.append(13)
    elif i > 2240 and i < 2401:
        training_y.append(14)
    else:
        print "Done with elements of training_y"

cross_test_y = []
for i in range(601):
    if i > 0 and i < 41:
        cross_test_y.append(0)
    elif i > 40 and i < 81:
        cross_test_y.append(1)
    elif i>80 and i<121:
        cross_test_y.append(2)
    elif i>120 and i<161:
        cross_test_y.append(3)
    elif i>160 and i<201:
        cross_test_y.append(4)
    elif i>200 and i<241:
        cross_test_y.append(5)
    elif i > 240 and i < 281:
        cross_test_y.append(6)
    elif i > 280 and i < 321:
        cross_test_y.append(7)
    elif i > 320 and i < 361:
        cross_test_y.append(8)
    elif i >360 and i < 401:
        cross_test_y.append(9)
    elif i > 400 and i < 441:
        cross_test_y.append(10)
    elif i > 440 and i < 481:
        cross_test_y.append(11)
    elif i > 480 and i < 521:
        cross_test_y.append(12)
    elif i > 520 and i <561:
        cross_test_y.append(13)
    elif i > 560 and i < 601:
        cross_test_y.append(14)
    else:
        print "Done with elements of cross_test_y"
y_out = []

#
# for i in data_train1:
#     print i
print "len of data_train1"
print len(data_train1)
print len(training_y)
print "training y"
br = 0
for i in training_y:
    if i == 14:
        br = br + 1
print br

print "len of data_test1"
print len(data_test1)
print len(cross_test_y)

for file in data_train5:
    img = cv2.imread("Renamed_Data/" + file)
    res=cv2.resize(img,(64,64))
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray_image, 127, 255,
                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    xarr=np.squeeze(np.array(thresh1).astype(np.float32))
    m,v=cv2.PCACompute(xarr, np.mean(xarr, axis=0).reshape(1,-1))
    arr= np.array(v)
    flat_arr= arr.ravel()
    training_set.append(flat_arr)

for file in data_test5:
    img = cv2.imread("Renamed_Data/" + file)
    res = cv2.resize(img, (64, 64))
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray_image, 127, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    xarr = np.squeeze(np.array(thresh1).astype(np.float32))
    m, v = cv2.PCACompute(xarr, np.mean(xarr, axis=0).reshape(1, -1))
    arr = np.array(v)
    flat_arr = arr.ravel()
    testing_set.append(flat_arr)
#
trainData=np.float32(training_set)
responses=np.float32(training_y)
print "Done with new_listing"

crossTrain=np.float32(trainData)
#
model = SVM(C=50, gamma=0.018)
model.train(crossTrain, np.array(training_y))

#
testData = np.float32(testing_set)
crossTest=np.float32(testData)
y_out = model.predict(crossTest)

total = 0
for x in xrange(600):
    if y_out[x] == cross_test_y[x]:
        total += 1

print "Percentage is " + str((total/600.0)*100) + "%"

#data_train1&data_test1 --> Percentage is 75.6260434057%
#data_train1&data_test2--> Percentage is 79.6666666667%
#data_train1&data_test3--->Percentage is 79.8333333333%
#data_train1&data_test4-->Percentage is 81.5%
#data_train1&data_test5-->Percentage is 81.1666666667%
# average: 79.554%