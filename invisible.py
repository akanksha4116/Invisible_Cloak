import cv2
import time
import numpy as np



cap=cv2.VideoCapture(0)
time.sleep(3)
background=0

for i in range(60):
    capture,background=cap.read()
background = np.flip(background,axis=1)


## Read every frame from the webcam, until the camera is open
while(cap.isOpened()):
    res,img=cap.read()
    if not res:
        
        break
    img = np.flip(img, axis = 1)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red= np.array([0,120,70])
    upper_red = np.array([10, 255,255])
    mask1=cv2.inRange(hsv,lower_red,upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask1 = mask1 + mask2
     ## Open and Dilate the mask image
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    ## Create an inverted mask to segment out the red color from the frame
    mask2 = cv2.bitwise_not(mask1)

    ## Segment the red color part out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(img, img, mask  =mask2)

    ## Create image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(background, background, mask=mask1)

    ## Generating the final output and writing
    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
    print(finalOutput)
    cv2.imshow("magic", finalOutput)
    k=cv2.waitKey(1)
    if k == ord('q'): 
        break


cap.release()
cv2.destroyAllWindows()