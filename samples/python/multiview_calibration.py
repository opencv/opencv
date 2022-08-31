import sys, traceback, cv2 as cv, numpy as np, os, json, argparse, matplotlib.pyplot as plt

def getDimBox(pts):
    return np.array([[pts[...,k].min(), pts[...,k].max()] for k in range(pts.shape[-1])])

def plotCamerasPosition(R, t, image_sizes):
    cam_box = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]],dtype=np.float32); cam_box[:,2] = 3.0
    cam_box *= 0.15
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax_lines = [None for i in range(len(R))]
    ax.set_title('Cameras position', fontsize=20)
    all_pts = []
    colors = np.random.RandomState(0).rand(len(R),3)
    for i, cam in enumerate(R):
        cam_box_i = cam_box.copy()
        cam_box_i[:,0] *= image_sizes[i][0] / max(image_sizes[i][1], image_sizes[i][0])
        cam_box_i[:,1] *= image_sizes[i][1] / max(image_sizes[i][1], image_sizes[i][0])
        cam_box_Rt = (R[i] @ cam_box_i.T + t[i]).T
        all_pts.append(np.concatenate((cam_box_Rt, t[i].T)))

        ax_lines[i] = ax.plot([t[i][0,0], cam_box_Rt[0,0]], [t[i][1,0], cam_box_Rt[0,1]], [t[i][2,0], cam_box_Rt[0,2]], '-', color=colors[i])[0]
        ax.plot([t[i][0,0], cam_box_Rt[1,0]], [t[i][1,0], cam_box_Rt[1,1]], [t[i][2,0], cam_box_Rt[1,2]], '-', color=colors[i])
        ax.plot([t[i][0,0], cam_box_Rt[2,0]], [t[i][1,0], cam_box_Rt[2,1]], [t[i][2,0], cam_box_Rt[2,2]], '-', color=colors[i])
        ax.plot([t[i][0,0], cam_box_Rt[3,0]], [t[i][1,0], cam_box_Rt[3,1]], [t[i][2,0], cam_box_Rt[3,2]], '-', color=colors[i])

        ax.plot([cam_box_Rt[0,0], cam_box_Rt[1,0]], [cam_box_Rt[0,1], cam_box_Rt[1,1]], [cam_box_Rt[0,2], cam_box_Rt[1,2]], '-', color=colors[i])
        ax.plot([cam_box_Rt[1,0], cam_box_Rt[2,0]], [cam_box_Rt[1,1], cam_box_Rt[2,1]], [cam_box_Rt[1,2], cam_box_Rt[2,2]], '-', color=colors[i])
        ax.plot([cam_box_Rt[2,0], cam_box_Rt[3,0]], [cam_box_Rt[2,1], cam_box_Rt[3,1]], [cam_box_Rt[2,2], cam_box_Rt[3,2]], '-', color=colors[i])
        ax.plot([cam_box_Rt[3,0], cam_box_Rt[0,0]], [cam_box_Rt[3,1], cam_box_Rt[0,1]], [cam_box_Rt[3,2], cam_box_Rt[0,2]], '-', color=colors[i])

    ax.legend(ax_lines, [str(i) for i in range(len(R))], fontsize=10)
    dim_box = getDimBox(np.concatenate((all_pts)))
    ax.set_xlim(dim_box[0]); ax.set_ylim(dim_box[1]); ax.set_zlim(dim_box[2])
    ax.set_box_aspect((dim_box[0, 1] - dim_box[0, 0], dim_box[1, 1] - dim_box[1, 0], dim_box[2, 1] - dim_box[2, 0]))
    ax.set_xlabel('x', fontsize=20); ax.set_ylabel('y', fontsize=20); ax.set_zlabel('z', fontsize=20)
    ax.view_init(azim=90, elev=-40)

def plotProjection(points_2d, pattern_points, rvec0, tvec0, rvec1, tvec1, K, dist_coeff, is_fisheye, image_size, image=None):
    rvec2, tvec2 = cv.composeRT(rvec0, tvec0, rvec1, tvec1)[:2]
    if is_fisheye:
        points_2d_est = cv.fisheye.projectPoints(pattern_points, rvec2, tvec2, K, dist_coeff.flatten())[0]
    else:
        points_2d_est = cv.projectPoints(pattern_points, rvec2, tvec2, K, dist_coeff)[0].reshape(-1,2)

    fig = plt.figure()
    if image is None:
        ax = fig.add_subplot(111)
        ax.scatter(points_2d[:,0], points_2d[:,1], color='red', marker='.')
        ax.scatter(points_2d_est[:,0], points_2d_est[:,1], color='green', marker='+')
        for pt1, pt2 in zip(points_2d, points_2d_est):
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='blue')
        dim_box = getDimBox(points_2d)
        ax.set_aspect('equal', 'box')
        # ax.set_xlim(0, image_size[0]); ax.set_ylim(0, image_size[1])
        ax.set_xlabel('x', fontsize=20); ax.set_ylabel('y', fontsize=20)
        ax.legend(['given', 'back-projected'], fontsize=15)
        ax.set_title('Comparison of given points and back-projected', fontsize=17)
    else:
        circle_sz = 5
        for pt1, pt2, in zip(points_2d, points_2d_est):
            cv.circle(image, np.array(pt1, dtype=int), circle_sz, (0, 0, 255), -1)
            cv.circle(image, np.array(pt2, dtype=int), circle_sz, (0, 255, 0), -1)
            cv.line(image, np.array(pt1, dtype=int), np.array(pt2, dtype=int), (255, 0, 0), 1)
        plt.imshow(image)

def calibrateFromPoints(pattern_points, image_points, image_sizes, is_fisheye, image_names=None):
    """
    pattern_points: NUM_POINTS x 3 (numpy array)
    image_points: NUM_CAMERAS x NUM_FRAMES x NUM_POINTS x 2
    is_fisheye: NUM_CAMERAS (bool)
    image_sizes: NUMCAMERAS x [width, height]
    """

    num_cameras = len(image_points)
    num_frames = len(image_points[0])
    visibility = np.zeros((num_cameras, num_frames), dtype=int)
    pattern_points_all = [pattern_points for i in range(num_frames)]
    # pattern_points_all = [pattern_points] * num_frames
    for i in range(num_cameras):
        for j in range(num_frames):
            visibility[i,j] = int(len(image_points[i][j]) != 0)

    success, rvecs, Ts, Ks, distortions, rvecs0, tvecs0, errors_per_frame, output_pairs = \
        cv.calibrateMultiview(objPoints=pattern_points_all,
                              imagePoints=image_points,
                              imageSize=image_sizes,
                              visibility=visibility,
                              #Ks=Ks,
                              #distortions=distortions,
                              is_fisheye=np.array(is_fisheye, dtype=int),
                              USE_INTRINSICS_GUESS=False,
                              flags_intrinsics=0)
    Rs = [cv.Rodrigues(rvec)[0] for rvec in rvecs]
    print('rvecs', Rs)
    print('tvecs', Ts)
    print('K', Ks)
    print('distortion', distortions)
    print('mean RMS error over all visible frames %.3E' % errors_per_frame[errors_per_frame > 0].mean())

    visibility_idxs = np.stack(np.where(visibility)) # 2 x M, first row is camera idx, second is frame idx
    plotCamerasPosition(Rs, Ts, image_sizes)
    def plot(cam_idx, frame_idx):
        image = None if image_names is None else cv.cvtColor(cv.imread(image_names[cam_idx][frame_idx]), cv.COLOR_BGR2RGB)
        plotProjection(image_points[cam_idx][frame_idx], pattern_points, rvecs0[frame_idx],
            tvecs0[frame_idx], rvecs[cam_idx], Ts[cam_idx], Ks[cam_idx], distortions[cam_idx],
            is_fisheye[cam_idx], image_sizes[cam_idx], image)
    plot(visibility_idxs[0,0], visibility_idxs[1,0])
    plt.show()

def chessboard_points(grid_size, dist_m):
    pattern = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
    pattern[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)*dist_m # only for (x,y,z=0)
    return pattern

def circles_grid_points(grid_size, dist_m):
    pattern = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            pattern.append([j*dist_m, i*dist_m, 0]);
    return np.array(pattern, dtype=np.float32)

def asym_circles_grid_points(grid_size, dist_m):
    pattern = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            pattern.append([(2*j + i % 2)*dist_m, i*dist_m, 0]);
    return np.array(pattern, dtype=np.float32)

def calibrateFromImages(files_with_images, grid_size, pattern_type, is_fisheye, dist_m, RESIZE_IMAGE=True):
    """
    files_with_images: NUM_CAMERAS x NUM_FRAMES x string - path to image file
    grid_size: [width, height] -- size of grid pattern
    dist_m: length of a grid cell
    is_fisheye: NUM_CAMERAS (bool)
    """
    if pattern_type.lower() == 'checkerboard':
        pattern = chessboard_points(grid_size, dist_m)
    elif pattern_type.lower() == 'circles':
        pattern = circles_grid_points(grid_size, dist_m)
    elif pattern_type.lower() == 'acircles':
        pattern = asym_circles_grid_points(grid_size, dist_m)
    else:
        raise "Pattern type is not implemented!"

    assert len(files_with_images) == len(is_fisheye) and len(grid_size) == 2
    image_points_cameras = []
    image_sizes = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    all_images_names = []
    for filename in files_with_images:
        images_names = open(filename, 'r').readlines()
        for i in range(len(images_names)):
            images_names[i] = images_names[i].replace('\n', '')
        all_images_names.append(images_names)
        image_points_camera = []
        img_size = None
        for img_name in images_names:
            print(img_name)
            if img_name == '': # above we replace new line '\n' with ''
                image_points_camera.append(np.array([], dtype=np.float32))
                continue

            assert os.path.exists(img_name)
            img = cv.cvtColor(cv.imread(img_name), cv.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = img.shape[:2][::-1]

            scale = 1.0
            window = (6,6)
            img_detection = img
            if RESIZE_IMAGE:
                scale = 1000.0 / max(img.shape[0], img.shape[1])
                if scale < 1.0:
                    img_detection = cv.resize(img, (int(scale * img.shape[1]), int(scale * img.shape[0])), interpolation=cv.INTER_AREA)
                    # increase refinement window for the original image resolution
                    window = (16, 16)

            if pattern_type.lower() == 'checkerboard':
                ret, corners = cv.findChessboardCorners(img_detection, grid_size, None)
            elif pattern_type.lower() == 'circles':
                ret, corners = cv.findCirclesGrid(img_detection, patternSize=grid_size, flags=cv.CALIB_CB_SYMMETRIC_GRID);
            elif pattern_type.lower() == 'acircles':
                ret, corners = cv.findCirclesGrid(img_detection, patternSize=grid_size, flags=cv.CALIB_CB_ASYMMETRIC_GRID);
            else:
                raise "Calibration pattern is not supported!"

            if ret:
                if scale < 1.0:
                    corners /= scale
                corners2 = cv.cornerSubPix(img, corners, window, (-1,-1), criteria)
                image_points_camera.append(np.array(corners2, dtype=np.float32).reshape(-1,2))
            else:
                image_points_camera.append(np.array([], dtype=np.float32))

        image_points_cameras.append(image_points_camera)
        image_sizes.append(img_size)
    calibrateFromPoints(pattern, image_points_cameras, image_sizes, is_fisheye, all_images_names)

def calibrateFromJSON(json_file):
    assert os.path.exists(json_file)
    data = json.load(open(json_file, 'r'))
    for i in range(len(data['image_points'])):
        for j in range(len(data['image_points'][i])):
            data['image_points'][i][j] = np.array(data['image_points'][i][j], dtype=np.float32)
    calibrateFromPoints(np.array(data['object_points'], dtype=np.float32).T, data['image_points'], data['image_sizes'], data['is_fisheye'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default=None, help="json file with all data. Must have keys: 'object_points', 'image_points', 'image_sizes', 'is_fisheye'")
    parser.add_argument('--filenames', type=str, default=None, help='files containg images, e.g., file1,file2,...,fileN for N cameras')
    parser.add_argument('--pattern_size', type=str, default=None, help='pattern size: width,height')
    parser.add_argument('--pattern_type', type=str, default=None, help='currently supported only checkeboard')
    parser.add_argument('--fisheye', type=str, default=None, help='fisheye mask, e.g., 0,1,...')
    parser.add_argument('--pattern_distance', type=float, default=None, help='distance between object / pattern points')
    params, _ = parser.parse_known_args()
    if params.json_file is not None:
        calibrateFromJSON(params.json_file)
    else:
        if (params.fisheye is None and params.filenames is None and params.pattern_type is None and \
                params.pattern_size is None and params.pattern_distance is None):
            assert False and 'Either json file or all other parameters must be set'
        calibrateFromImages(params.filenames.split(','), [int(v) for v in params.pattern_size.split(',')], 
            params.pattern_type, [bool(int(v)) for v in params.fisheye.split(',')], params.pattern_distance)
