import numpy as np
import cv2 as cv
import quaternion
import argparse

def make_Rt(val):
    q = np.quaternion(val[6], val[3], val[4] ,val[5])
    R = quaternion.as_rotation_matrix(q)
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
        help="""TSDF      - reconstruct data in volume with bounders,
                HashTSDF  - reconstruct data in volume without bounders (infinit volume),
                ColorTSDF - like TSDF and reconstruct colors too,
                defalt - runs TSDF""",
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

    volume_type = cv.TSDF
    if args.algo == "HashTSDF":
        volume_type = cv.HashTSDF
    elif args.algo == "ColorTSDF":
        volume_type = cv.ColorTSDF

    volume = cv.Volume(volume_type)

    for key in list(depth_info.keys())[:]:
        Rt = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        )
        for key1 in groundtruth_info:
            if np.abs(key1 - key) < 0.01:
                Rt = make_Rt(groundtruth_info[key1])
                break
        
        rgb_path = ''
        for key1 in rgb_info:
            if np.abs(key1 - key) < 0.05:
                rgb_path = path + rgb_info[key1]
                break

        depth = cv.imread(path + depth_info[key], cv.IMREAD_ANYDEPTH).astype(np.float32)
        rgb = cv.imread(rgb_path, cv.IMREAD_COLOR).astype(np.float32)

        if volume_type != cv.ColorTSDF:
            volume.integrate(depth, Rt)
        else:
            volume.integrate(depth, rgb, Rt)

        size = (480, 640, 4)
        
        points  = np.zeros(size, np.float32)
        normals = np.zeros(size, np.float32)
        colors = np.zeros(size, np.float32)
        
        if volume_type != cv.ColorTSDF:
            volume.raycast(Rt, points, normals)
        else:
            volume.raycast(Rt, size[0], size[1], points, normals, colors)
        
        x, y, z, zeros = cv.split(points)

        cv.imshow("X", np.absolute(x))
        cv.imshow("Y", np.absolute(y))
        cv.imshow("Z", z)
        
        if volume_type == cv.ColorTSDF:
            cv.imshow("Color", colors.astype(np.uint8))

        cv.waitKey(10)

if __name__ == '__main__':
  main()