from __future__ import print_function
import numpy as np
import cv2
from cv2 import dnn
import timeit

def prepare_image(img):
    img = cv2.resize(img, (224, 224))
    #convert interleaved image (RGBRGB) to planar(RRGGBB)
    blob = np.moveaxis(img, 2, 0)
    blob = np.reshape(blob.astype(np.float32), (-1, 3, 224, 224))
    return blob

def timeit_forward(net):
    print("OpenCL:", cv2.ocl.useOpenCL())
    print("Runtime:", timeit.timeit(lambda: net.forward(), number=10))

def get_class_list():
    with open('synset_words.txt', 'rt') as f:
        return [ x[x.find(" ") + 1 :] for x in f ]

blob = prepare_image(cv2.imread('space_shuttle.jpg'))
print("Input:", blob.shape, blob.dtype)

cv2.ocl.setUseOpenCL(True)  #Disable OCL if you want
net = dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')
net.setBlob(".data", blob)
net.forward()
#timeit_forward(net)        #Uncomment to check performance

prob = net.getBlob("prob")
print("Output:", prob.shape, prob.dtype)
classes = get_class_list()
print("Best match", classes[prob.argmax()])