import numpy as np
import cv2 as cv
import argparse

# Use source data from this site:
# https://vision.in.tum.de/data/datasets/rgbd-dataset/download
# For example if you use rgbd_dataset_freiburg1_xyz sequence, your prompt should be:
# python /path_to_opencv/samples/python/volume.py --source_folder /path_to_datasets/rgbd_dataset_freiburg1_xyz --algo <some algo>
# so that the folder contains files groundtruth.txt and depth.txt

# for more info about this function look cv::Quat::toRotMat3x3(...)
def quatToMat3(a, b, c, d):
    return np.array([
        [1 - 2 * (c * c + d * d), 2 * (b * c - a * d)    , 2 * (b * d + a * c)],
        [2 * (b * c + a * d)    , 1 - 2 * (b * b + d * d), 2 * (c * d - a * b)],
        [2 * (b * d - a * c)    , 2 * (c * d + a * b)    , 1 - 2 * (b * b + c * c)]
    ])

def make_Rt(val):
    R = quatToMat3(val[6], val[3], val[4] ,val[5])
    t = np.array([ [val[0]], [val[1]], [val[2]] ])
    tmp = np.array([0, 0, 0, 1])

    Rt = np.append(R, t , axis=1 )
    Rt = np.vstack([Rt, tmp])

    return Rt

def get_image_info(path, is_depth):
    image_info = {}
    source = 'depth.txt'
    if not is_depth:
        source = 'rgb.txt'
    with open(path+source) as file:
        lines = file.readlines()
        for line in lines:
            words = line.split(' ')
            if words[0] == '#':
                continue
            image_info[float(words[0])] = words[1][:-1]
    return image_info

def get_groundtruth_info(path):
    groundtruth_info = {}
    with open(path+'groundtruth.txt') as file:
        lines = file.readlines()
        for line in lines:
            words = line.split(' ')
            if words[0] == '#':
                continue
            groundtruth_info[float(words[0])] = [float(i) for i in words[1:]]
    return groundtruth_info

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algo',
        help="""TSDF      - reconstruct data in volume with bounds,
                HashTSDF  - reconstruct data in volume without bounds (infinite volume),
                ColorTSDF - like TSDF but also keeps color data,
                default - runs TSDF""",
        default="")
    parser.add_argument(
        '-src',
        '--source_folder',
        default="")

    args = parser.parse_args()

    path = args.source_folder
    if path[-1] != '/':
        path += '/'

    depth_info = get_image_info(path, True)
    rgb_info = get_image_info(path, False)
    groundtruth_info = get_groundtruth_info(path)

    volume_type = cv.VolumeType_TSDF
    if args.algo == "HashTSDF":
        volume_type = cv.VolumeType_HashTSDF
    elif args.algo == "ColorTSDF":
        volume_type = cv.VolumeType_ColorTSDF

    settings = cv.VolumeSettings(volume_type)
    volume = cv.Volume(volume_type, settings)

    for key in list(depth_info.keys())[:]:
        Rt = np.eye(4)
        for key1 in groundtruth_info:
            if np.abs(key1 - key) < 0.01:
                Rt = make_Rt(groundtruth_info[key1])
                break

        rgb_path = ''
        for key1 in rgb_info:
            if np.abs(key1 - key) < 0.05:
                rgb_path = path + rgb_info[key1]
                break

        depthPath = path + depth_info[key]
        depth = cv.imread(depthPath, cv.IMREAD_ANYDEPTH).astype(np.float32)
        if depth.size <= 0:
            raise Exception('Failed to load depth file: %s' % depthPath)

        rgb = cv.imread(rgb_path, cv.IMREAD_COLOR).astype(np.float32)
        if rgb.size <= 0:
            raise Exception('Failed to load RGB file: %s' % rgb_path)

        if volume_type != cv.VolumeType_ColorTSDF:
            volume.integrate(depth, Rt)
        else:
            volume.integrateColor(depth, rgb, Rt)

        size = (480, 640, 4)

        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)
        colors = np.zeros(size, np.float32)

        if volume_type != cv.VolumeType_ColorTSDF:
            volume.raycast(Rt, points, normals)
        else:
            volume.raycastColor(Rt, points, normals, colors)

        channels = list(cv.split(points))

        cv.imshow("X", np.absolute(channels[0]))
        cv.imshow("Y", np.absolute(channels[1]))
        cv.imshow("Z", channels[2])

        if volume_type == cv.VolumeType_ColorTSDF:
            cv.imshow("Color", colors.astype(np.uint8))

        #TODO: also display normals

        cv.waitKey(10)

if __name__ == '__main__':
    main()
