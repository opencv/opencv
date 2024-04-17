# Script is based on https://github.com/richzhang/colorization/blob/master/colorization/colorize.py
# To download the onnx model, see: https://storage.googleapis.com/ailia-models/colorization/colorizer.onnx
# python colorization.py --onnx_model_path colorizer.onnx --input ansel_adams3.jpg
import numpy as np
import argparse
import cv2 as cv
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='iColor: deep interactive colorization')
    parser.add_argument('--input', default='ansel_adams3.jpg',help='Path to image.')
    parser.add_argument('--onnx_model_path', help='Path to onnx model', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img=cv.imread(cv.samples.findFile(args.input))

    img_lab = cv.cvtColor(img, cv.COLOR_RGB2Lab)
    img_l = img_lab[:,:,0] # pull out L channel

    img_rgb_rs = cv.resize(img, (256, 256), interpolation=cv.INTER_CUBIC)
    img_lab_rs = cv.cvtColor(img_rgb_rs, cv.COLOR_BGR2Lab)

    # Optionally normalize Lab output to match skimage's scale
    img_lab_rs = img_lab_rs.astype(np.float32)  # Convert to float to avoid data overflow
    img_lab_rs[:, :, 0] *= (100.0 / 255.0)      # Scale L channel to 0-100 range
    img_l_rs = img_lab_rs[:,:,0]

    onnx_model_path = args.onnx_model_path  # Update this path to your ONNX model's path
    session = cv.dnn.readNetFromONNX(onnx_model_path)

    # Process each image in the batch (assuming batch processing is needed)
    img=img_l_rs.astype(np.float32)
    print(img.shape)
    blob = cv.dnn.blobFromImage(img, swapRB=False)  # Adjust swapRB according to your model's training
    print(blob.shape)
    session.setInput(blob)
    result_numpy = np.array(session.forward()[0])

    # Assume img_l is the original L channel with shape (H, W)
    # and result_numpy is the AB channels with shape (2, H, W).
    # First, correct the shape of result_numpy if needed:
    if result_numpy.shape[0] == 2:
        # Transpose result_numpy to shape (H, W, 2)
        ab = result_numpy.transpose((1, 2, 0))
    else:
        # If it's already (H, W, 2), assign it directly
        ab = result_numpy

    assert img_l.ndim == 2

    # Resize ab to match img_l's dimensions if they are not the same
    h, w = img_l.shape
    if ab.shape[:2] != (h, w):
        ab_resized = cv.resize(ab, (w, h), interpolation=cv.INTER_LINEAR)
    else:
        ab_resized = ab

    # Expand dimensions of L to match ab's dimensions
    img_l_expanded = np.expand_dims(img_l, axis=-1)

    # Concatenate L with AB to get the LAB image
    lab_image = np.concatenate((img_l_expanded, ab_resized), axis=-1)

    # Convert the Lab image to a 32-bit float format
    lab_image = lab_image.astype(np.float32)

    # Normalize L channel to the range [0, 100] and AB channels to the range [-127, 127]
    lab_image[:, :, 0] *= (100.0 / 255.0)  # Rescale L channel
    #lab_image[:, :, 1:] -= 128              # Shift AB channels

    # Convert the LAB image to BGR
    image_bgr_out = cv.cvtColor(lab_image, cv.COLOR_Lab2BGR)

    cv.imshow("output image",image_bgr_out)
    cv.waitKey(0)