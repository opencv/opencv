import opencv

capture = opencv.VideoCapture(0)
img = opencv.Mat()

while True:
    capture.read(img)
    opencv.imshow("camera",img)
    if opencv.waitKey(10) == 27:
        break
