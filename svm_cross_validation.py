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

for i in range(1,201):
    listing1.append("c"+format(i))
for i in range(1,201):
    listing1.append("d"+format(i))
for i in range(1,201):
    listing1.append("f"+format(i))
for i in range(1,201):
    listing1.append("five"+format(i))
for i in range(1,201):
    listing1.append("four"+format(i))
for i in range(1,201):
    listing1.append("g"+format(i))
for i in range(1,201):
    listing1.append("i"+format(i))
for i in range(1,201):
    listing1.append("l"+format(i))
for i in range(1,201):
    listing1.append("nine"+format(i))
for i in range(1,201):
    listing1.append("one"+format(i))
for i in range(1,201):
    listing1.append("r"+format(i))
for i in range(1,201):
    listing1.append("three"+format(i))
for i in range(1,201):
    listing1.append("two"+format(i))
for i in range(1,201):
    listing1.append("u"+format(i))
for i in range(1,201):
    listing1.append("v"+format(i))

new_listing = []
for i in range(3000):
    new_listing.append(listing1[i]+".png")

training_set = []
testing_set = []
training_y = []
for i in range(3000):
    if i > 0 and i < 201:
        training_y.append(0)
    elif i > 200 and i < 401:
        training_y.append(1)
    elif i>400 and i<601:
        training_y.append(2)
    elif i>600 and i<801:
        training_y.append(3)
    elif i>800 and i<1001:
        training_y.append(4)
    elif i>1000 and i<1201:
        training_y.append(5)
    elif i > 1200 and i < 1401:
        training_y.append(6)
    elif i > 1400 and i < 1601:
        training_y.append(7)
    elif i > 1600 and i < 1801:
        training_y.append(8)
    elif i >1800 and i < 2001:
        training_y.append(9)
    elif i > 2000 and i < 2201:
        training_y.append(10)
    elif i > 2200 and i < 2401:
        training_y.append(11)
    elif i > 2400 and i < 2601:
        training_y.append(12)
    elif i > 2600 and i <2801:
        training_y.append(13)
    elif i > 2800 and i < 3001:
        training_y.append(14)
y_out = []

for file in new_listing:
    if file != "l121.png":
        img = cv2.imread("Renamed_Data/" + file)
        res=cv2.resize(img,(64,64))
        gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        xarr=np.squeeze(np.array(gray_image).astype(np.float32))
        m,v=cv2.PCACompute(xarr, np.mean(xarr, axis=0).reshape(1,-1))
        arr= np.array(v)
        flat_arr= arr.ravel()
        training_set.append(flat_arr)

trainData=np.float32(training_set)
counter = 1
cross_train_x = []
cross_test_x = []
cross_train_y = []
cross_test_y = []

for i in trainData:
    if counter == 15:
        cross_test_x.append(i)
        counter = 1
    else:
        cross_train_x.append(i)
        counter = counter + 1

counter = 1
for i in training_y:
    if counter == 15:
        cross_test_y.append(i)
        counter = 1
    else:
        cross_train_y.append(i)
        counter = counter + 1

print "Done with elements of cross_train_x and cross_train_y"

crossData=np.float32(cross_train_x)
responses=np.float32(cross_train_y)

model = SVM(C=2, gamma=0.027)
model.train(crossData, np.array(cross_train_y))

print "Done with training"

testData = np.float32(cross_test_x)

y_out = model.predict(testData)


total = 0
for i in xrange(199):
    if y_out[i] == cross_test_y[i]:
        total += 1

print "Percentage is " + str((total/199.0)*100) + "%"

