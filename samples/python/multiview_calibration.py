import pdb, sys, traceback, cv2 as cv, numpy as np, os, json, argparse

def calibrateFromPoints(pattern_points, image_points, image_sizes, is_fisheye):
    """
    pattern_points: NUM_POINTS x 3
    image_points: NUM_CAMERAS x NUM_FRAMES x NUM_POINTS x 2
    is_fisheye: NUM_CAMERAS (bool)
    image_sizes: NUMCAMERAS x [width, height]
    """

    num_cameras = len(image_points)
    num_frames = len(image_points[0])
    visibility = np.zeros((num_cameras, num_frames), dtype=bool)
    pattern_points_all = [pts for pts in pattern_points]
    for i in range(num_cameras):
        for j in range(num_frames):
            visibility[i,j] = len(image_points[i][j]) != 0

    success, Rs, Ts, Ks, distortions, rvecs0, tvecs0, errors_per_frame, output_pairs = cv.calibrateMultiview(pattern_points_all,
                image_points, image_sizes, visibility, is_fisheye, cv.USE_INTRINSICS_GUESS=False)
    print(Rs)
    print(Ts)
    print(Ks)
    print(distortions)

def calibrateFromImages(files_with_images, grid_size, is_fisheye, dist_m):
    """
    files_with_images: NUM_CAMERAS x NUM_FRAMES x string - path to image file
    grid_size: [width, height] -- size of grid pattern
    dist_m: length of a grid cell
    is_fisheye: NUM_CAMERAS (bool)
    """
    pattern = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
    pattern[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)*dist_m # only for (x,y,z=0)

    image_points_cameras = []
    image_sizes = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    for filename in files_with_images:
        images_names = open(filename, 'r').readlines()
        image_points_camera = []
        img_size = None
        for img_name in images_names:
            assert os.path.exists(img_name)
            img = cv.imread(img_name)
            if img_size is None:
                img_size = img.shape[:2][::-1]
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            window = (11,11)
            scale = min(1.0, 1000 / max(img.shape[0], img.shape[1]))
            if scale < 1.0:
                gray = cv.resize(gray, (int(scale * gray.shape[1]), int(scale * gray.shape[0])), interpolation=cv.INTER_AREA)
                window = (16, 16) # increase refinement window

            ret, corners = cv.findChessboardCorners(gray, grid_size, None)
            if ret:
                if scale < 1.0: corners /= scale
                corners2 = cv.cornerSubPix(gray, corners, window, (-1,-1), criteria)
                image_points_camera.append([corners2])
            else:
                image_points_camera.append([])
        image_points_cameras.append(image_points_camera)
        image_sizes.append(img_size)

    calibrateFromPoints(pattern, image_points_cameras, image_sizes, is_fisheye)

def calibrateFromJSON(json_file):
    assert os.path.exists(json_file)
    data = json.load(open(json_file, 'r'))
    calibrateFromPoints(data['pattern'], data['image_points'], data['image_sizes'], data['is_fisheye'])

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--json_file', type=str)
        params, _ = parser.parse_known_args()
        calibrateFromJSON(params.json_file)
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)    