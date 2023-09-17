import cv2 as cv

def boundaries_rect(pt1, pt2):
    xmin, xmax = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
    ymin, ymax = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
    return (xmin, ymin), (xmax, ymax)

def pt_in_rect(pt, tl, br):
    return tl[0] <= pt[0] <= br[0] and tl[1] <= pt[1] <= br[1]

img = cv.imread(cv.samples.findFile('butterfly.jpg'))
pt1 = (30, 40)
pt2 = (200, 90)
h = cv.LineIterator(pt1, pt2)

pt = h.pos()
pt_tl, pt_br = boundaries_rect(pt1, pt2)
while pt_in_rect(pt, pt_tl, pt_br):
    img[pt[1], pt[0]] = 255
    h.next()
    pt = h.pos()
cv.imshow("Line", img)
cv.waitKey()
pt1 = (40, 80)
pt2 = (30, 200)
h = cv.LineIterator(pt1, pt2)

pt = h.pos()
pt_tl, pt_br = boundaries_rect(pt1, pt2)
while pt_in_rect(pt, pt_tl, pt_br):
    img[pt[1], pt[0]] = 255
    h.next()
    pt = h.pos()
cv.imshow("Line", img)
cv.waitKey()
h = cv.LineIterator((33, 85, 7, 10), pt1, pt2)
pt = h.pos()
pt_tl, pt_br = boundaries_rect(pt1, pt2)
while pt_in_rect(pt, pt_tl, pt_br):
    img[pt[1], pt[0]] = [0, 0, 255]
    pt = h.next()
cv.imshow("Line", img)
cv.waitKey()

cv.destroyAllWindows()
