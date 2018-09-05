import cv2
import numpy as np

def nothing(x):
    pass

def trackBar():
    # Create a black image, a window
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('Image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'Image', 0, 255, nothing)
    cv2.createTrackbar('G', 'Image', 0, 255, nothing)
    cv2.createTrackbar('B', 'Image', 0, 255, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'Image', 0, 1, nothing)

    while(1):
        cv2.imshow('Image', img)
        k = cv2.waitKey(1) 
        if k == 27:  #27 is the code for ESC key
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'Image')
        g = cv2.getTrackbarPos('G', 'Image')
        b = cv2.getTrackbarPos('B', 'Image')
        s = cv2.getTrackbarPos(switch, 'Image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b,g,r]

    cv2.destroyAllWindows()
    
trackBar()