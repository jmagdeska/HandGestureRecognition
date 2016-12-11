from sklearn.svm import SVC
import cv2
import numpy as np
import math
from PIL import Image

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

def black_pixels(filename):
        #im = Image.open(filename)
        im = cv2.imread(filename)
        nblack = 0
        BLACK_MIN = np.array([0, 0, 0], np.uint8)
        BLACK_MAX = np.array([0, 0, 1], np.uint8)

        pixel = cv2.inRange(im, BLACK_MIN, BLACK_MAX)
        nblack = cv2.countNonZero(pixel)
        #print "black pixels" + format(nblack)
        return nblack

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
#print "Elements of the listing1 are:    "
for i in range(3000):
    #print listing1[i]
    new_listing.append(listing1[i]+".png")

training_set =  np.empty((2999,4,))
testing_set = np.empty((15,4,))
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
    else:
        print "Done with elements of training_y"

y_out = []
#za sega nepotrebno
#y_correct = [2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0]
t = 0
for file in new_listing:
    if file != "l121.png":
        img = cv2.imread("Renamed_Data/" + file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new, contours, hierarchy = cv2.findContours(gray.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        points = []
        num_fingers = 0

        for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                points.append(far)
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                if angle <= 90:
                    num_fingers += 1

        black_pix = black_pixels("Renamed_Data/" + file)
        width, height = Image.open("Renamed_Data/" + file).size
        all_pixels = width * height
        white_pix = all_pixels - black_pix
        ratio = white_pix/black_pix
        training_set[t][0] = num_fingers
        training_set[t][1] = black_pix
        training_set[t][2] = white_pix
        training_set[t][3] = ratio
        t = t + 1
        

trainData=np.float32(training_set)
responses=np.float32(training_y)
print "Done with new_listing"

cross_train_x = np.zeros((2800,4))
cross_test_x = np.zeros((199,4))
cross_train_y = []
cross_test_y = []

test_counter = 0
train_counter = 0
counter = 1

for i in trainData:
    if counter == 15:
        cross_test_x[test_counter][:] = i
        test_counter = test_counter + 1
        counter = 1
    else:
        cross_train_x[train_counter][:] = i
        train_counter = train_counter + 1
        counter = counter + 1

counter = 1
for i in training_y:
    if counter == 15:
        cross_test_y.append(i)
        counter = 1
    else:
        cross_train_y.append(i)
        counter = counter + 1

crossTrain=np.float32(cross_train_x)

model = SVM(C=5, gamma=0.029)
model.train(crossTrain, np.array(cross_train_y))

t = 0
for i in xrange(1,16):
    img = cv2.imread("test_data/test" + str(i) + ".png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new, contours, hierarchy = cv2.findContours(gray.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    points = []
    num_fingers = 0

    for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            points.append(far)
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                num_fingers += 1

    black_pix = black_pixels("test_data/test" + str(i) + ".png")
    width, height = Image.open("test_data/test" + str(i) + ".png").size
    all_pixels = width * height
    white_pix = all_pixels - black_pix
    ratio = white_pix/black_pix
    testing_set[t][0] = num_fingers
    testing_set[t][1] = black_pix
    testing_set[t][2] = white_pix
    testing_set[t][3] = ratio

    t = t + 1

print "Done with test_data"
testData = np.float32(testing_set)
y_out = model.predict(testData)

print "RESULTS"

for y in y_out:
    if y == 0.0:
        print "Letter: C"
    elif y == 1.0:
        print "Letter D"
    elif y == 2.0:
        print "Letter F"
    elif y == 3.0:
        print "Number: five"
    elif y == 4.0:
        print "Number: four"
    elif y == 5.0:
        print "Letter G"
    elif y == 6.0:
        print "Letter I"
    elif y == 7.0:
        print "Letter L"
    elif y == 8.0:
        print "Number: nine"
    elif y == 9.0:
        print "Number: one"
    elif y == 10.0:
        print "Letter R"
    elif y == 11.0:
        print "Number: three"
    elif y == 12.0:
        print "Number: two"
    elif y == 13.0:
        print "Letter: U"
    elif y == 14.0:
        print "Letter: V"
    else:
       print "Error"

crossTestY = [6, 7, 7, 6, 0, 12, 7, 6, 6, 15, 5, 8, 4, 7, 9]

total = 0
for x in xrange(15):
    if y_out[x] == crossTestY[x]:
        total += 1

print y_out
print "Percentage is " + str((total/15.0)*100) + "%"