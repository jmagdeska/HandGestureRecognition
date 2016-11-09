from sklearn.svm import SVC
import cv2
import numpy as np

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

listing1 = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10",
            "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9", "l10",
            "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"]
training_set = []
testing_set = []
training_y = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
y_out = []
y_correct = [2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0]

for file in listing1:
 img = cv2.imread("nova_data/" + file + ".png")
 res=cv2.resize(img,(64,64))
 gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
 xarr=np.squeeze(np.array(gray_image).astype(np.float32))
 m,v=cv2.PCACompute(xarr, np.mean(xarr, axis=0).reshape(1,-1))
 arr= np.array(v)
 flat_arr= arr.ravel()
 training_set.append(flat_arr)

trainData=np.float32(training_set)
responses=np.float32(training_y)

model = SVM(C=2.67, gamma=0.01)
model.train(trainData, np.array(training_y))

for i in xrange(1,10):
 img = cv2.imread("test_data/test" + str(i) + ".png")
 res=cv2.resize(img,(64,64))
 gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
 xarr=np.squeeze(np.array(gray_image).astype(np.float32))
 m,v=cv2.PCACompute(xarr, np.mean(xarr, axis=0).reshape(1,-1))
 arr= np.array(v)
 flat_arr= arr.ravel()
 testing_set.append(flat_arr)

testData = np.float32(testing_set)
y_out = model.predict(testData)

for y in y_out:
    if y == 0.0:
        print "Letter: C"
    elif y == 1.0:
        print "Letter L"
    elif y == 2.0:
        print "Letter I"

total = 0
for i in xrange(9):
    if y_out[i] == y_correct[i]:
        total += 1

print "Percentage is " + str((total/9.0)*100) + "%"

