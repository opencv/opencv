# coding: utf-8

import sys, os, glob

CAFFE_ROOT = "/home/vitaliy/opencv/caffe/"
sys.path.insert(0, CAFFE_ROOT + 'python')

import numpy as np
import caffe
#import cv2

def get_cafe_output(inp_blob, proto_name, caffemodel_name):
    caffe.set_mode_cpu()
    net = caffe.Net(proto_name, caffe.TEST)

    #net.blobs['input'].reshape(*inp_blob.shape)
    net.blobs['input'].data[...] = inp_blob

    net.forward()
    out_blob = net.blobs['output'].data[...];

    if net.params.get('output'):
        print "Params count:", len(net.params['output'])
        net.save(caffemodel_name)

    return out_blob

if __name__ == '__main__':
    proto_filenames = glob.glob("layer_*.prototxt")

    for proto_filename in proto_filenames:
        proto_filename = os.path.basename(proto_filename)
        proto_basename = os.path.splitext(proto_filename)[0]
        cfmod_basename = proto_basename + ".caffemodel"
        npy_filename = proto_basename + ".npy"

        inp_blob_name = proto_basename + ".input.npy"
        inp_blob = np.load(inp_blob_name) if os.path.exists(inp_blob_name) else np.load('blob.npy')

        print "\nGenerate data for:"
        print cfmod_basename, inp_blob.shape

        out_blob = get_cafe_output(inp_blob, proto_filename, cfmod_basename)
        print out_blob.shape
        np.save(npy_filename, out_blob)
