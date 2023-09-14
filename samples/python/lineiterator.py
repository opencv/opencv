import cv2 as cv

img = cv.imread(cv.samples.findFile('butterfly.jpg'))
pt1 = (30, 40)
pt2 = (200, 90)
h = cv.LineIterator(pt1, pt2)

pt = h.pos()
while min(pt1[0], pt2[0]) <= pt[0] <= max(pt1[0], pt2[0]) and \
      min(pt1[1], pt2[1]) <= pt[1] <= max(pt1[1], pt2[1]):
    img[pt[1], pt[0]] = 255
    h.next()
    pt = h.pos()
cv.imshow("Line", img)
cv.waitKey()
pt1 = (40, 80)
pt2 = (30, 200)
h = cv.LineIterator(pt1, pt2)

pt = h.pos()
while min(pt1[0], pt2[0]) <= pt[0] <= max(pt1[0], pt2[0]) and \
      min(pt1[1], pt2[1]) <= pt[1] <= max(pt1[1], pt2[1]):
    img[pt[1], pt[0]] = 255
    pt = h.next()
cv.imshow("Line", img)
cv.waitKey()
h = cv.LineIterator((33, 85, 7, 10), pt1, pt2)
pt = h.pos()
while min(pt1[0], pt2[0]) <= pt[0] <= max(pt1[0], pt2[0]) and \
      min(pt1[1], pt2[1]) <= pt[1] <= max(pt1[1], pt2[1]):
    img[pt[1], pt[0]] = [0, 0, 255]
    pt = h.next()
cv.imshow("Line", img)
cv.waitKey()

cv.destroyAllWindows()
