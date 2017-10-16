# Script is based on https://github.com/richzhang/colorization/blob/master/colorize.py
# To download the caffemodel and the prototxt, see: https://github.com/richzhang/colorization/tree/master/models
# To download pts_in_hull.npy, see: https://github.com/richzhang/colorization/blob/master/resources/pts_in_hull.npy
import numpy as np
import argparse
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser(description='iColor: deep interactive colorization')
    parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--prototxt', help='Path to colorization_deploy_v2.prototxt', default='./models/colorization_release_v2.prototxt')
    parser.add_argument('--caffemodel', help='Path to colorization_release_v2.caffemodel', default='./models/colorization_release_v2.caffemodel')
    parser.add_argument('--kernel', help='Path to pts_in_hull.npy', default='./resources/pts_in_hull.npy')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    W_in = 224
    H_in = 224
    imshowSize = (640, 480)

    args = parse_args()

    # Select desired model
    net = cv.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)

    pts_in_hull = np.load(args.kernel) # load cluster centers

    # populate cluster centers as 1x1 convolution kernel
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    if args.input:
        cap = cv.VideoCapture(args.input)
    else:
        cap = cv.VideoCapture(0)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)

        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
        img_l = img_lab[:,:,0] # pull out L channel
        (H_orig,W_orig) = img_rgb.shape[:2] # original image size

        # resize image to network input size
        img_rs = cv.resize(img_rgb, (W_in, H_in)) # resize image to network input size
        img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
        img_l_rs = img_lab_rs[:,:,0]
        img_l_rs -= 50 # subtract 50 for mean-centering

        net.setInput(cv.dnn.blobFromImage(img_l_rs))
        ab_dec = net.forward('class8_ab')[0,:,:,:].transpose((1,2,0)) # this is our result

        (H_out,W_out) = ab_dec.shape[:2]
        ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
        img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

        frame = cv.resize(frame, imshowSize)
        cv.imshow('origin', frame)
        cv.imshow('gray', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        cv.imshow('colorized', cv.resize(img_bgr_out, imshowSize))
