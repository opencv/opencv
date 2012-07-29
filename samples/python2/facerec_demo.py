#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Philipp Wagner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the author nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import os
import sys

import PIL.Image as Image

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2

def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].    
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    
    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 
    
    Returns:
        A list [X,y]
        
            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

def create_font(fontname='Tahoma', fontsize=10):
    """Creates a font for the subplot."""
    return { 'fontname': fontname, 'fontsize':fontsize }

def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
    """This will ease creating a subplot with matplotlib a lot for us."""
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center') 
    for i in xrange(len(images)):
        ax0 = fig.add_subplot(rows,cols,(i+1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

def imsave(image, title="", filename=None):
    """Saves or shows (if no filename is given) an image."""
    fig = plt.figure()
    plt.imshow(np.asarray(image))
    plt.title(title, create_font('Tahoma',10))
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
    
if __name__ == "__main__":
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) != 2:
        print "USAGE: facerec_demo.py </path/to/images>"
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X,y] = read_images(sys.argv[1])
    # Create the Eigenfaces model. We are going to use the default
    # parameters for this simple example, please read the documentation
    # for thresholding:
    model = cv2.createEigenFaceRecognizer()
    # Read
    # Learn the model. Remember our function returns Python lists,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
    model.train(np.asarray(X), np.asarray(y))
    # We now get a prediction from the model! In reality you 
    # should always use unseen images for testing your model. 
    # But so many people were confused, when I sliced an image 
    # off in the C++ version, so I am just using an image we 
    # have trained with.
    #
    # model.predict is going to return the predicted label and
    # the associated confidence:
    [p_label, p_confidence] = model.predict(np.asarray(X[0]))
    # Print it:
    print "Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence)
    # Cool! Finally we'll plot the Eigenfaces, because that's 
    # what most people read in the papers are keen to see.
    #
    # Just like in C++ you have access to all model internal
    # data, because the cv::FaceRecognizer is a cv::Algorithm.
    #
    # You can see the available parameters with getParams():
    print model.getParams()
    # Now let's get some data:
    mean = model.getMat("mean")
    eigenvectors = model.getMat("eigenvectors")
    # We'll save the mean, by first normalizing it:
    mean_norm = normalize(mean, 0, 255)
    mean_resized = mean_norm.reshape(X[0].shape)
    imsave(mean_resized, "Mean Face", "mean.png")
    # Turn the first (at most) 16 eigenvectors into grayscale 
    # images. You could also use cv::normalize here, but sticking
    # to NumPy is much easier for now. 
    # Note: eigenvectors are stored by column:
    SubplotData = []
    for i in xrange(min(len(X), 16)):
        eigenvector_i = eigenvectors[:,i].reshape(X[0].shape)
        SubplotData.append(normalize(eigenvector_i, 0, 255))
    # Plot them and store the plot to "python_eigenfaces.png"
    subplot(title="Eigenfaces AT&T Facedatabase", images=SubplotData, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="eigenfaces.png")
