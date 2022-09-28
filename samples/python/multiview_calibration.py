import sys, traceback, cv2 as cv, numpy as np, os, json, argparse, matplotlib.pyplot as plt, time, joblib, multiprocessing

def getDimBox(pts):
    return np.array([[pts[...,k].min(), pts[...,k].max()] for k in range(pts.shape[-1])])

def plotCamerasPosition(R, t, image_sizes, pairs, pattern, frame_idx):
    cam_box = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]],dtype=np.float32)
    cam_box[:,2] = 3.0
    dist_to_pattern = np.linalg.norm(pattern.mean(0))
    cam_box *= 0.1*dist_to_pattern
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax_lines = [None for i in range(len(R))]
    ax.set_title('Cameras position and pattern of frame '+str(frame_idx), loc='center', wrap=True, fontsize=20)
    all_pts = [pattern]
    colors = np.random.RandomState(0).rand(len(R),3)
    for i in range(len(R)):
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

    for (i,j) in pairs:
        edge_line = ax.plot([t[i][0,0], t[j][0,0]], [t[i][1,0], t[j][1,0]], [t[i][2,0], t[j][2,0]], '-', color='black')[0]
    ax.scatter(pattern[:,0], pattern[:,1], pattern[:,2], color='red', marker='o')
    ax.legend(ax_lines+[edge_line], [str(i) for i in range(len(R))]+['stereo pair'], fontsize=10)
    dim_box = getDimBox(np.concatenate((all_pts)))
    ax.set_xlim(dim_box[0]); ax.set_ylim(dim_box[1]); ax.set_zlim(dim_box[2])
    ax.set_box_aspect((dim_box[0, 1] - dim_box[0, 0], dim_box[1, 1] - dim_box[1, 0], dim_box[2, 1] - dim_box[2, 0]))
    ax.set_xlabel('x', fontsize=20); ax.set_ylabel('y', fontsize=20); ax.set_zlabel('z', fontsize=20)
    ax.view_init(azim=90, elev=-40)

def plotProjection(points_2d, pattern_points, rvec0, tvec0, rvec1, tvec1, K, dist_coeff, is_fisheye, cam_idx, frame_idx, per_acc, image=None):
    rvec2, tvec2 = cv.composeRT(rvec0, tvec0, rvec1, tvec1)[:2]
    if is_fisheye:
        points_2d_est = cv.fisheye.projectPoints(pattern_points[:,None], rvec2, tvec2, K, dist_coeff.flatten())[0].reshape(-1,2)
    else:
        points_2d_est = cv.projectPoints(pattern_points, rvec2, tvec2, K, dist_coeff)[0].reshape(-1,2)
    fig = plt.figure()
    errs = np.linalg.norm(points_2d - points_2d_est, axis=-1)
    mean_err = errs.mean()
    title = "Comparison of given point (start) and back-projected (end). Cam. "+str(cam_idx)+" frame "+\
            str(frame_idx)+" mean err. (px) %.1f"%mean_err+". In top %.0f"%per_acc+"% accurate frames"
    dist_pattern = np.linalg.norm(points_2d_est.min(0) - points_2d_est.max(0))
    width = 2e-3*dist_pattern
    head_width = 5*width
    if image is None:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
    else:
        plt.imshow(image)
        ax = plt.gca()
    num_colors = 8
    cmap_fnc = lambda x : np.concatenate((x, 1-x, np.zeros_like(x)))
    cmap = cmap_fnc(np.linspace(0,1,num_colors)[None,:])
    thrs = np.linspace(0, 10, num_colors)
    arrows = [None for i in range(num_colors)]
    for k, (pt1, pt2) in enumerate(zip(points_2d, points_2d_est)):
        color = cmap[:,-1]
        for i in range(len(thrs)):
            if errs[k] < thrs[i]:
                color = cmap[:,i]
                break
        arrow = ax.arrow(pt1[0], pt1[1], pt2[0]-pt1[0], pt2[1]-pt1[1], color=color, width=width, head_width=head_width)
        for i in range(len(thrs)):
            if errs[k] < thrs[i]:
                arrows[i] = arrow
                break
    legend, legend_str = [], []
    for i in range(num_colors):
        if arrows[i] is not None:
            legend.append(arrows[i])
            if i == 0:
                legend_str.append('lower than %.1f'%thrs[i])
            elif i == num_colors-1:
                legend_str.append('higher than %.1f'%thrs[i])
            else:
                legend_str.append('between %.1f'%thrs[i-1]+' and %.1f'%thrs[i])
    ax.legend(legend, legend_str, fontsize=15)
    ax.set_title(title, loc='center', wrap=True, fontsize=16)

def calibrateFromPoints(pattern_points, image_points, image_sizes, is_fisheye, image_names=None, find_intrinsics_in_advance=False, Ks=None, distortions=None):
    """
    pattern_points: NUM_POINTS x 3 (numpy array)
    image_points: NUM_CAMERAS x NUM_FRAMES x NUM_POINTS x 2
    is_fisheye: NUM_CAMERAS (bool)
    image_sizes: NUMCAMERAS x [width, height]
    """

    num_cameras = len(image_points)
    num_frames = len(image_points[0])
    visibility = np.zeros((num_cameras, num_frames), dtype=int)
    pattern_points_all = [pattern_points] * num_frames
    for i in range(num_cameras):
        for j in range(num_frames):
            visibility[i,j] = int(len(image_points[i][j]) != 0)
    with np.printoptions(threshold=np.inf):
        print("Visibility Matrix:\n", np.transpose(visibility))

    if Ks is not None and distortions is not None:
        USE_INTRINSICS_GUESS = True
    else:
        USE_INTRINSICS_GUESS = find_intrinsics_in_advance
        if find_intrinsics_in_advance:
            Ks, distortions = [], []
            for c in range(num_cameras):
                image_points_c = [image_points[c][f] for f in range(num_frames) if len(image_points[c][f]) > 0]
                repr_err_c, K, dist_coeff, _, _ = cv.calibrateCamera([pattern_points] * len(image_points_c), image_points_c, image_sizes[c], None, None)
                print('intrinsics calibration for camera', c, ', reproj error %.2f (px)' % repr_err_c)
                Ks.append(K)
                distortions.append(dist_coeff)

    start_time = time.time()
    success, rvecs, Ts, Ks, distortions, rvecs0, tvecs0, errors_per_frame, output_pairs = \
        cv.calibrateMultiview(objPoints=pattern_points_all,
                              imagePoints=image_points,
                              imageSize=image_sizes,
                              visibility=visibility,
                              Ks=Ks,
                              distortions=distortions,
                              is_fisheye=np.array(is_fisheye, dtype=int),
                              USE_INTRINSICS_GUESS=USE_INTRINSICS_GUESS,
                              flags_intrinsics=0)
    print('calibration time', time.time() - start_time, 'seconds')
    assert success
    Rs = [cv.Rodrigues(rvec)[0] for rvec in rvecs]
    print('rvecs', Rs)
    print('tvecs', Ts)
    print('K', Ks)
    print('distortion', distortions)
    errors = errors_per_frame[errors_per_frame > 0]
    print('mean RMS error over all visible frames %.3E' % errors.mean())
    with np.printoptions(precision=2):
        print('mean RMS errors per camera', np.array([np.mean(errs[errs > 0]) for errs in errors_per_frame]))

    visibility_idxs = np.stack(np.where(visibility)) # 2 x M, first row is camera idx, second is frame idx
    frame_idx = visibility_idxs[1,0]
    R_frame = cv.Rodrigues(rvecs0[frame_idx])[0]
    pattern_frame = (R_frame @ pattern_points.T + tvecs0[frame_idx]).T
    plotCamerasPosition(Rs, Ts, image_sizes, output_pairs, pattern_frame, frame_idx)
    def plot(cam_idx, frame_idx):
        image = None if image_names is None else cv.cvtColor(cv.imread(image_names[cam_idx][frame_idx]), cv.COLOR_BGR2RGB)
        plotProjection(image_points[cam_idx][frame_idx], pattern_points, rvecs0[frame_idx],
            tvecs0[frame_idx], rvecs[cam_idx], Ts[cam_idx], Ks[cam_idx], distortions[cam_idx],
            is_fisheye[cam_idx], cam_idx, frame_idx, 1e2*(errors_per_frame[cam_idx,frame_idx]<errors).sum()/len(errors), image)
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
            pattern.append([j*dist_m, i*dist_m, 0])
    return np.array(pattern, dtype=np.float32)

def asym_circles_grid_points(grid_size, dist_m):
    pattern = []
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            if i % 2 == 1:
                pattern.append([(j+.5)*dist_m, dist_m*(i//2+.5), 0])
            else:
                pattern.append([j*dist_m, (i//2)*dist_m, 0])
    return np.array(pattern, dtype=np.float32)

def detect(cam_idx, frame_idx, img_name, pattern_type, grid_size, criteria, RESIZE_IMAGE):
    print(img_name)
    assert os.path.exists(img_name)
    img = cv.cvtColor(cv.imread(img_name), cv.COLOR_BGR2GRAY)
    img_size = img.shape[:2][::-1]

    scale = 1.0
    window = (5,5)
    img_detection = img
    if RESIZE_IMAGE:
        scale = 1000.0 / max(img.shape[0], img.shape[1])
        if scale < 1.0:
            img_detection = cv.resize(img, (int(scale * img.shape[1]), int(scale * img.shape[0])), interpolation=cv.INTER_AREA)

    if pattern_type.lower() == 'checkerboard':
        ret, corners = cv.findChessboardCorners(img_detection, grid_size, None)
    elif pattern_type.lower() == 'circles':
        ret, corners = cv.findCirclesGrid(img_detection, patternSize=grid_size, flags=cv.CALIB_CB_SYMMETRIC_GRID)
    elif pattern_type.lower() == 'acircles':
        ret, corners = cv.findCirclesGrid(img_detection, patternSize=grid_size, flags=cv.CALIB_CB_ASYMMETRIC_GRID)
    else:
        raise "Calibration pattern is not supported!"

    if ret:
        if scale < 1.0:
            corners /= scale
        corners2 = cv.cornerSubPix(img, corners, window, (-1,-1), criteria)
        # cv.drawChessboardCorners(img, grid_size, corners2, ret)
        # plt.imshow(img)
        # plt.show()
        return cam_idx, frame_idx, img_size, np.array(corners2, dtype=np.float32).reshape(-1,2)
    else:
        return cam_idx, frame_idx, img_size, np.array([], dtype=np.float32)

def calibrateFromImages(files_with_images, grid_size, pattern_type, is_fisheye, dist_m, RESIZE_IMAGE=True, find_intrinsics_in_advance=False, is_parallel_detection=False):
    """
    files_with_images: NUM_CAMERAS - path to file containing image names (NUM_FRAMES)
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
    all_images_names, input_data = [], []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    for cam_idx, filename in enumerate(files_with_images):
        assert os.path.exists(filename)
        images_names = open(filename, 'r').readlines()
        for i in range(len(images_names)):
            images_names[i] = images_names[i].replace('\n', '')
        all_images_names.append(images_names)
        if cam_idx > 0:
            # same number of images per file
            assert len(images_names) == len(all_images_names[-1])
        for frame_idx, img_name in enumerate(images_names):
            input_data.append([cam_idx, frame_idx, img_name])

    image_sizes = [None for i in range(len(files_with_images))]
    image_points_cameras = [[None for j in range(len(images_names))] for i in range(len(files_with_images))]
    if is_parallel_detection:
        output = joblib.Parallel(n_jobs=multiprocessing.cpu_count()) \
            (joblib.delayed(detect)(cam_idx, frame_idx, img_name, pattern_type, grid_size, criteria, RESIZE_IMAGE) for cam_idx, frame_idx, img_name in input_data)
        for cam_idx, frame_idx, img_size, corners in output:
            image_points_cameras[cam_idx][frame_idx] = corners
            if image_sizes[cam_idx] is None:
                image_sizes[cam_idx] = img_size
    else:
        for cam_idx, frame_idx, img_name in input_data:
            _, _, img_size, corners = detect(cam_idx, frame_idx, img_name, pattern_type, grid_size, criteria, RESIZE_IMAGE)
            image_points_cameras[cam_idx][frame_idx] = corners
            if image_sizes[cam_idx] is None:
                image_sizes[cam_idx] = img_size
    calibrateFromPoints(pattern, image_points_cameras, image_sizes, is_fisheye, all_images_names, find_intrinsics_in_advance)

def calibrateFromJSON(json_file, find_intrinsics_in_advance=False):
    assert os.path.exists(json_file)
    data = json.load(open(json_file, 'r'))
    for i in range(len(data['image_points'])):
        for j in range(len(data['image_points'][i])):
            data['image_points'][i][j] = np.array(data['image_points'][i][j], dtype=np.float32)
    Ks = data['Ks'] if 'Ks' in data else None
    distortions = data['distortions'] if 'distortions' in data else None
    images_names = data['images_names'] if 'images_names' in data else None
    calibrateFromPoints(np.array(data['object_points'], dtype=np.float32).T, data['image_points'], data['image_sizes'],
        data['is_fisheye'], images_names, find_intrinsics_in_advance, Ks, distortions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default=None, help="json file with all data. Must have keys: 'object_points', 'image_points', 'image_sizes', 'is_fisheye'")
    parser.add_argument('--filenames', type=str, default=None, help='files containg images, e.g., file1,file2,...,fileN for N cameras')
    parser.add_argument('--pattern_size', type=str, default=None, help='pattern size: width,height')
    parser.add_argument('--pattern_type', type=str, default=None, help='supported: checkeboard, circles, acircles')
    parser.add_argument('--fisheye', type=str, default=None, help='fisheye mask, e.g., 0,1,...')
    parser.add_argument('--pattern_distance', type=float, default=None, help='distance between object / pattern points')
    parser.add_argument('--find_intrinsics_in_advance', type=int, default=0, help='calibrate intrinsics in advance, 0 - False, 1 - True')
    params, _ = parser.parse_known_args()
    if params.json_file is not None:
        calibrateFromJSON(params.json_file, params.find_intrinsics_in_advance==1)
    else:
        if (params.fisheye is None and params.filenames is None and params.pattern_type is None and \
                params.pattern_size is None and params.pattern_distance is None):
            assert False and 'Either json file or all other parameters must be set'
        calibrateFromImages(params.filenames.split(','), [int(v) for v in params.pattern_size.split(',')],
            params.pattern_type, [bool(int(v)) for v in params.fisheye.split(',')], params.pattern_distance,
            find_intrinsics_in_advance=params.find_intrinsics_in_advance==1)
