'''
Author Gourav Roy
email gouravroy261999[at]gmail.com
usage python black_and_white_to_color_image.py --prototxt <path to protxt> --model <path to caffe model> --points <path to NumPy cluster center points >  --image <path to imput image>
ne.jpeg

'''


import numpy as np
import cv2 as cv


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True,help="path to input black and white image")
    parser.add_argument("-p", "--prototxt", type=str, required=True,help="path to Caffe prototxt file")
    parser.add_argument("-m", "--model", type=str, required=True,help="path to Caffe pre-trained model")
    parser.add_argument("-c", "--points", type=str, required=True,help="path to cluster center points")
    args = vars(parser.parse_args())
    # load our serialized black and white colorizer model and cluster
    # center points from disk
    print("[INFO] loading model...")
    net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    pts = np.load(args["points"])
    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # load the input image from disk, scale the pixel intensities to the
    # range [0, 1], and then convert the image from the BGR to Lab color
    # space
    image = cv.imread(args["image"])
    scaled = image.astype("float32") / 255.0
    lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization
    # network accepts), split channels, extract the 'L' channel, and then
    resized = cv.resize(lab, (224, 224))
    img_l = cv.split(resized)[0]

    #subtracting 50 for mean centring
    img_l -= 50

    # pass the L channel through the network which will *predict* the 'a'
    # and 'b' channel values
    'print("[INFO] colorizing image...")'
    net.setInput(cv.dnn.blobFromImage(img_l))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab = cv.resize(ab, (image.shape[1], image.shape[0]))
    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the
    # predicted 'ab' channels
    img_l = cv.split(lab)[0]
    colorized = np.concatenate((img_l[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then
    # clip any values that fall outside the range [0, 1]
    colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned
    # 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")

    # show the original and output colorized images
    cv.imshow("Original", image)
    cv.imshow("Colorized", colorized)
    cv.waitKey(0)
if __name__=="__main__":
    main()
