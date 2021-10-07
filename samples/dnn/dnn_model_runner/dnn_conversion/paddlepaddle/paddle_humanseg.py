import os
import paddlehub.vision.transforms as T
import numpy as np
import cv2 as cv


def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


def visualize(image, result, save_dir=None, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = get_color_map_list(256)
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv.LUT(result, color_map[:, 0])
    c2 = cv.LUT(result, color_map[:, 1])
    c3 = cv.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    im = cv.imread(image)
    vis_result = cv.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv.imwrite(out_path, vis_result)
    else:
        return vis_result


def preprocess(image_path):
    ''' preprocess input image file to np.ndarray

    Args:
        image_path(str): Path of input image file

    Returns:
        ProcessedImage(numpy.ndarray): A numpy.ndarray
                variable which shape is (1, 3, 192, 192)
    '''
    transforms = T.Compose([
        T.Resize((192, 192)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ],
        to_rgb=True)
    return np.expand_dims(transforms(image_path), axis=0)


if __name__ == '__main__':
    img_path = "../../../../data/messi5.jpg"
    # load PPSeg Model use cv.dnn
    net = cv.dnn.readNetFromONNX('humanseg_hrnet18_tiny.onnx')
    # read and preprocess image file
    im = preprocess(img_path)
    # inference
    net.setInput(im)
    result = net.forward(['save_infer_model/scale_0.tmp_1'])
    # post process
    image = cv.imread(img_path)
    r, c, _ = image.shape
    result = np.argmax(result[0], axis=1).astype(np.uint8)
    result = cv.resize(result[0, :, :],
                       dsize=(c, r),
                       interpolation=cv.INTER_NEAREST)

    print("grid_image.shape is: ", result.shape)
    folder_path = "data"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, '%s.jpg' % "result_test_human")
    result_color = visualize(img_path, result)
    cv.imwrite(file_path, result_color)
    print('%s saved' % file_path)
