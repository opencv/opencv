import cv2 as cv
import numpy as np

img = cv.imread(cv.samples.findFile('sudoku.png'))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
num_lines, _ = lines.shape
for i in range(num_lines):
    x1, y1, x2, y2 = lines[i]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imwrite('houghlines5.jpg',img)