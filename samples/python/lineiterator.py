import cv2 as cv


img = cv.imread(cv.samples.findFile('butterfly.jpg'))
pt1 = (30, 40)
pt2 = (40, 50)
h = cv.LineIterator(pt1, pt2)
print(h.pos(), h.iter)
for pt in h:
    print(pt, h.iter)
    img[pt[1], pt[0]] = 255
cv.imshow("Line", img)
cv.waitKey()

pt1 = (40, 50)
pt2 = (210, 100)
h = cv.LineIterator(pt1, pt2, 4)
pt = h.pos()
img[pt[1], pt[0]] = 255

while h.next():
    pt = h.pos()
    img[pt[1], pt[0]] = 255
cv.imshow("Line", img)
cv.waitKey()

pt1 = (40, 80)
pt2 = (30, 200)
h = cv.LineIterator(pt1, pt2)

pt = h.pos()
img[pt[1], pt[0]] = 255
while h.next():
    pt = h.pos()
    img[pt[1], pt[0]] = 255
cv.imshow("Line", img)
cv.waitKey()

cv.destroyAllWindows()
