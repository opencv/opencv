import cv2
import sys
import os

filename = sys.argv[1] if len(sys.argv) > 1 else "animated_image.webp"

# Load collections
collection1 = cv2.ImageCollection(filename, cv2.IMREAD_UNCHANGED)
collection2 = cv2.ImageCollection(filename, cv2.IMREAD_REDUCED_GRAYSCALE_2)
collection3 = cv2.ImageCollection(filename, cv2.IMREAD_REDUCED_COLOR_2)
collection4 = cv2.ImageCollection(filename, cv2.IMREAD_COLOR_RGB)

if collection1.getLastError():
    print("Failed to initialize ImageCollection")
    sys.exit(-1)

size = collection1.size()
width = collection1.getWidth()
height = collection1.getHeight()
type_info = collection1.getType()

print(f"size   : {size}")
print(f"width  : {width}")
print(f"height : {height}")
print(f"type   : {type_info}")

idx1 = idx2 = idx3 = 0

print("Controls:\n"
      "  a/d: prev/next idx1\n"
      "  j/l: prev/next idx2\n"
      "  z/c: prev/next idx3\n"
      "  ESC or q: exit")

while True:
    cv2.imshow("Image 1", collection1.at(idx1))
    cv2.imshow("Image 2", collection2.at(idx2))
    cv2.imshow("Image 3", collection3.at(idx3))
    cv2.imshow("Image 4", collection4.at(idx1))

    key = cv2.waitKey(0)

    if key == ord('a'):
        idx1 -= 1
    elif key == ord('d'):
        idx1 += 1
    elif key == ord('j'):
        idx2 -= 1
    elif key == ord('l'):
        idx2 += 1
    elif key == ord('z'):
        idx3 -= 1
    elif key == ord('c'):
        idx3 += 1
    elif key in (ord('q'), 27):  # ESC or q
        break

    idx1 = max(0, min(idx1, collection1.size() - 1))
    idx2 = max(0, min(idx2, collection2.size() - 1))
    idx3 = max(0, min(idx3, collection3.size() - 1))

cv2.destroyAllWindows()
