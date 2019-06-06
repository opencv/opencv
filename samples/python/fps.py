import cv2
import time

# cv2.VideoCapture(0) gives return as True 
# but then print(frame) gives me an array filles with zeros

# cv2.VideoCapture(cv2.CAP_DSHOW) works just fine in this case

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
fps = cap.get(cv2.CAP_PROP_FPS)

ref = 60

#for images

while True:
	start = time.time()

	for i in range(0, ref):
		ret, frame = cap.read()

	end = time.time()
	number_of_seconds = end-start
	fps=ref/number_of_seconds
	print(fps)
	k=cv2.waitKey(0)
	if k==27:
		break

'''
# for videos
while True:
    ret, frame = cap.read()
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    k=cv2.waitKey(0)
	if k==27:
		break
'''
        
cap.release()
cv2.destroyAllWindows()
