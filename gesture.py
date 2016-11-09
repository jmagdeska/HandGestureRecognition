import cv2
import numpy as np
import math
import Tkinter as tk
import Tkinter as ttk
NORM_FONT= ("Verdana", 10)

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

i=1
list = []
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, img = cap.read()

    if not ret:
        break
    k = cv2.waitKey(1)
    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
    crop_img = img[50:400, 50:400]
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)
    # da go zacuva threshot vo datoteka so soodvetni dimenzii slicni so tie vo data_training
    msg = "done"
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        #SPACE is pressed
        #cv2.imwrite("screen" + format(i) + ".png", thresh1)
        cv2.imwrite("name" + format(len(list)+1) + ".png", thresh1)
        list.append("name" + format(len(list)+1) + ".png")
        popupmsg(msg)

    (version, _, _) = cv2.__version__.split('.')

    if version is '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version is '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
    cv2.drawContours(drawing,[hull],0,(0,0,255),0)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img,far,1,[0,0,255],-1)
        dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(crop_img,start,end,[0,255,0],2)
        cv2.circle(crop_img,far,5,[0,0,255],-1)
    if count_defects == 1:
        cv2.putText(img,"One more try", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        str = "hand gesture recognizer"
        cv2.putText(img, str, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 3:
        cv2.putText(img,"try again", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 4:
        cv2.putText(img,"Smile and wave!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    else:
        cv2.putText(img,"Place your hand in the rectangle", (30,30),\
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    #cv2.imshow('drawing', drawing)
    #cv2.imshow('end', crop_img)
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    #cv2.imshow('Contours', all_img)
    k = cv2.waitKey(10)
    if k == 27:
        break

