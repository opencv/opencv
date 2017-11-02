from __future__ import print_function
import numpy as np
import cv2
from cv2 import dnn
import timeit

def timeit_forward(net):
    print("Runtime:", timeit.timeit(lambda: net.forward(), number=10))

def get_class_list():
    with open('synset_words.txt', 'rt') as f:
        return [x[x.find(" ") + 1:] for x in f]

blob = dnn.blobFromImage(cv2.imread('space_shuttle.jpg'), 1, (224, 224), (104, 117, 123), False)
print("Input:", blob.shape, blob.dtype)

net = dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')
net.setInput(blob)
prob = net.forward()
#timeit_forward(net)        #Uncomment to check performance

print("Output:", prob.shape, prob.dtype)
classes = get_class_list()
print("Best match", classes[prob.argmax()])