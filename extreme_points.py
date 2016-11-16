# USAGE
# python extreme_points.py

# import the necessary packages
import cv2

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("screenshots/name2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest
# one
(version, _, _) = cv2.__version__.split('.')
if version is '3':
	image, cnts, hierarchy = cv2.findContours(thresh.copy(), \
		   cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
elif version is '2':
	cnts, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE, \
		   cv2.CHAIN_APPROX_NONE)

cnts = cnts[0]
c = max(cnts, key=cv2.contourArea)

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# draw the outline of the object, then draw each of the
# extreme points, where the left-most is red, right-most
# is green, top-most is blue, and bottom-most is teal
cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 6, (0, 0, 255), -1)
cv2.circle(image, extRight, 6, (0, 255, 0), -1)
cv2.circle(image, extTop, 6, (255, 0, 0), -1)
cv2.circle(image, extBot, 6, (255, 255, 0), -1)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)