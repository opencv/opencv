# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

# The script generates synthetic data for multi-camera calibration assessment
# Input: cameras configuration. See config_cv_test.yaml
# Output: generated object points (3d), image points (2d) for calibration and
#         board poses ground truth (R, t) for check

import argparse
import numpy as np
import math
import yaml
from drawer import animation2D, animation3D
from utils import RandGen, insideImage, eul2rot, saveKDRT, areAllInsideImage, insideImageMask, projectCamera, export2JSON, writeMatrix
from pathlib import Path
from board import CheckerBoard
import os
import json

class Camera:
    def __init__(self, idx, img_width, img_height, fx_limit, euler_limit, t_limit, is_fisheye, fy_deviation=None, skew=None,
                distortion_limit=None, noise_scale_img_diag=None):
        """
        @skew : is either None or in radians
        @fy_deviation : is either None (that is fx=fy) or value such that fy = [fx*(1-fy_deviation/100), fx*(1+fy_deviation/100)]
        @distortion_limit : is either None or array of size (num_tangential_dist+num_radial_dist) x 2
        @euler_limit : is 3 x 2 limit of euler angles in degrees
        @t_limit : is 3 x 2 limit of translation in meters
        """
        assert len(fx_limit) == 2 and img_width >= 0 and img_width >= 0
        if is_fisheye and distortion_limit is not None: assert len(distortion_limit) == 4 # distortion for fisheye has only 4 parameters
        self.idx = idx
        self.img_width, self.img_height = img_width, img_height
        self.fx_min = fx_limit[0]
        self.fx_max = fx_limit[1]
        self.fy_deviation = fy_deviation
        self.img_diag = math.sqrt(img_height ** 2 + img_width ** 2)
        self.is_fisheye = is_fisheye
        self.fx, self.fy = None, None
        self.px, self.py = None, None
        self.K, self.R, self.t, self.P = None, None, None, None
        self.skew = skew
        self.distortion = None
        self.distortion_lim = distortion_limit
        self.euler_limit = np.array(euler_limit, dtype=np.float32)
        self.t_limit = t_limit
        self.noise_scale_img_diag = noise_scale_img_diag
        if idx != 0:
            assert len(euler_limit) == len(t_limit) == 3
            for i in range(3):
                assert len(euler_limit[i]) == len(t_limit[i]) == 2
                self.euler_limit[i] *= (np.pi / 180)

def generateAll(cameras, board, num_frames, rand_gen, MAX_RAND_ITERS=10000, save_proj_animation=None, save_3d_animation=None):
    EPS = 1e-10
    """
    output:
        points_2d: NUM_FRAMES x NUM_CAMERAS x 2 x NUM_PTS
    """

    for i in range(len(cameras)):
        cameras[i].t = np.zeros((3, 1))
        if cameras[i].idx == 0:
            cameras[i].R = np.identity(3)
        else:
            angles = [0, 0, 0]
            for k in range(3):
                if abs(cameras[i].t_limit[k][0] - cameras[i].t_limit[k][1]) < EPS:
                    cameras[i].t[k] = cameras[i].t_limit[k][0]
                else:
                    cameras[i].t[k] = rand_gen.randRange(cameras[i].t_limit[k][0], cameras[i].t_limit[k][1])

                if abs(cameras[i].euler_limit[k][0] - cameras[i].euler_limit[k][1]) < EPS:
                    angles[k] = cameras[i].euler_limit[k][0]
                else:
                    angles[k] = rand_gen.randRange(cameras[i].euler_limit[k][0], cameras[i].euler_limit[k][1])

            cameras[i].R = eul2rot(angles)

        if abs(cameras[i].fx_min - cameras[i].fx_max) < EPS:
            cameras[i].fx = cameras[i].fx_min
        else:
            cameras[i].fx = rand_gen.randRange(cameras[i].fx_min, cameras[i].fx_max)
        if cameras[i].fy_deviation is None:
            cameras[i].fy = cameras[i].fx
        else:
            cameras[i].fy = rand_gen.randRange((1 - cameras[i].fy_deviation) * cameras[i].fx,
                                      (1 + cameras[i].fy_deviation) * cameras[i].fx)

        cameras[i].px = int(cameras[i].img_width / 2.0) + 1
        cameras[i].py = int(cameras[i].img_height / 2.0) + 1
        cameras[i].K = np.array([[cameras[i].fx, 0, cameras[i].px], [0, cameras[i].fy, cameras[i].py], [0, 0, 1]], dtype=float)
        if cameras[i].skew is not None: cameras[i].K[0, 1] = np.tan(cameras[i].skew) * cameras[i].K[0, 0]
        cameras[i].P = cameras[i].K @ np.concatenate((cameras[i].R, cameras[i].t), 1)

        if cameras[i].distortion_lim is not None:
            cameras[i].distortion = np.zeros((1, len(cameras[i].distortion_lim))) # opencv using 5 values distortion as default
            for k, lim in enumerate(cameras[i].distortion_lim):
                cameras[i].distortion[0,k] = rand_gen.randRange(lim[0], lim[1])
        else:
            cameras[i].distortion = np.zeros((1, 5)) # opencv is using 5 values distortion as default

    origin = None
    box = np.array([[0, board.square_len * (board.w - 1), 0, board.square_len * (board.w - 1)],
                    [0, 0, board.square_len * (board.h - 1), board.square_len * (board.h - 1)],
                    [0, 0, 0, 0]])

    if board.t_origin is None:
        try:
            import torch, pytorch3d, pytorch3d.transforms
            has_pytorch = True
        except:
            has_pytorch = False

        if has_pytorch:
            rot_angles = torch.zeros(3, requires_grad=True)
            origin = torch.ones((3,1), requires_grad=True)
            optimizer = torch.optim.Adam([rot_angles, origin], lr=5e-3)
            Ps = torch.tensor(np.stack([cam.K @ np.concatenate((cam.R, cam.t), 1) for cam in cameras]), dtype=torch.float32)
            rot_conv = 'XYZ'
            board_pattern = torch.tensor(box, dtype=Ps.dtype)
            corners = torch.tensor([[[0, 0], [0, cam.img_height], [cam.img_width, 0], [cam.img_width, cam.img_height]] for cam in cameras], dtype=Ps.dtype).transpose(-1,-2)
            loss_fnc = torch.nn.HuberLoss()
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-4, factor=0.8, patience=10)
            prev_loss = 1e10
            torch.autograd.set_detect_anomaly(True)
            MAX_DEPTH = 4
            for it in range(500):
                pts_board = pytorch3d.transforms.euler_angles_to_matrix(rot_angles, rot_conv) @ board_pattern + origin
                pts_proj = Ps[:,:3,:3] @ pts_board[None,:] + Ps[:,:,[-1]]
                pts_proj = pts_proj[:, :2] / (pts_proj[:, [2]]+1e-15)

                loss = num_wrong = 0
                for i, proj in enumerate(pts_proj):
                    if not areAllInsideImage(pts_proj[i], cameras[i].img_width, cameras[i].img_height):
                        loss += loss_fnc(corners[i], pts_proj[i])
                        num_wrong += 1
                if num_wrong > 0:
                    loss /= num_wrong
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step(loss)
                    if origin[2] < 0:
                        with torch.no_grad(): origin[2] = 2.0
                    if it % 5 == 0:
                        print('iter', it, 'loss %.2E' % loss)
                        if abs(prev_loss - loss) < 1e-10:
                            break
                        prev_loss = loss.item()
                else:
                    print('all points inside')
                    break
            print(origin)
            points_board = (torch.tensor(board.pattern, dtype=Ps.dtype) + origin).detach().numpy()
        else:
            max_sum_diag = 0.0
            total_tested = 0
            for z in np.arange(0.25, 50, .5):
                if origin is not None: break  # will not update
                min_x1, max_x1 = -z * cameras[0].px / cameras[0].fx, (cameras[0].img_width * z - z * cameras[0].px) / cameras[0].fx
                min_y1, max_y1 = -z * cameras[0].py / cameras[0].fy, (cameras[0].img_height * z - z * cameras[0].py) / cameras[0].fy
                min_x2, max_x2 = -z * cameras[0].px / cameras[0].fx - box[0, 1], (cameras[0].img_width * z - z * cameras[0].px) / cameras[0].fx - box[0, 1]
                min_y2, max_y2 = -z * cameras[0].py / cameras[0].fy - box[1, 2], (cameras[0].img_height * z - z * cameras[0].py) / cameras[0].fy - box[1, 2]
                min_x = max(min_x1, min_x2)
                min_y = max(min_y1, min_y2)
                max_x = min(max_x1, max_x2)
                max_y = min(max_y1, max_y2)
                if max_x < min_x or max_y < min_y: continue
                for x in np.linspace(min_x, max_x, 40):
                    for y in np.linspace(min_y, max_y, 40):
                        total_tested += 1
                        pts = box + np.array([[x], [y], [z]])
                        sum_diag = 0.0
                        all_visible = True
                        for i in range(len(cameras)):
                            pts_proj = projectCamera(cameras[i], pts)
                            visible_pts = insideImage(pts_proj, cameras[i].img_width, cameras[i].img_height)
                            if visible_pts != pts_proj.shape[1]:
                                # print(i,')',x, y, z, 'not visible, total', visible_pts, '/', pts_proj.shape[1])
                                all_visible = False
                                break
                            sum_diag += np.linalg.norm(pts_proj[:, 0] - pts_proj[:, -1])
                        if not all_visible: continue
                        if max_sum_diag < sum_diag:
                            max_sum_diag = sum_diag
                            origin = np.array([[x], [y], [z]])
            points_board = board.pattern + origin
    else:
        points_board = board.pattern + board.t_origin

    points_2d, points_3d = [], []
    valid_frames_per_camera = np.zeros(len(cameras))
    MIN_FRAMES_PER_CAM = int(num_frames * 0.1)
    R_used = []
    t_used = []
    for frame in range(MAX_RAND_ITERS):
        R_board = eul2rot([ rand_gen.randRange(board.euler_limit[0][0], board.euler_limit[0][1]),
                            rand_gen.randRange(board.euler_limit[1][0], board.euler_limit[1][1]),
                            rand_gen.randRange(board.euler_limit[2][0], board.euler_limit[2][1])])
        t_board = np.array([[rand_gen.randRange(board.t_limit[0][0], board.t_limit[0][1])],
                            [rand_gen.randRange(board.t_limit[1][0], board.t_limit[1][1])],
                            [rand_gen.randRange(board.t_limit[2][0], board.t_limit[2][1])]])

        points_board_mean = points_board.mean(-1)[:,None]
        pts_board = R_board @ (points_board - points_board_mean) + points_board_mean + t_board
        cam_points_2d = [projectCamera(cam, pts_board) for cam in cameras]

        """
        # plot normals
        board_normal = 10*np.cross(pts_board[:,board.w] - pts_board[:,0], pts_board[:,board.w-1] - pts_board[:,0])
        ax = plotCamerasAndBoardFig(pts_board, cameras, pts_color=board.colors_board)
        pts = np.stack((pts_board[:,0], pts_board[:,0]+board_normal))
        ax.plot(pts[:,0], pts[:,1], pts[:,2], 'r-')
        for ii, cam in enumerate(cameras):
            pts = np.stack((cam.t.flatten(), cam.t.flatten()+cam.R[2]))
            ax.plot(pts[:,0], pts[:,1], pts[:,2], 'g-')
            print(ii, np.arccos(board_normal.dot(cam.R[2]) / np.linalg.norm(board_normal))*180/np.pi, np.arccos((-board_normal).dot(cam.R[2]) / np.linalg.norm(board_normal))*180/np.pi)
        plotAllProjectionsFig(np.stack(cam_points_2d), cameras, pts_color=board.colors_board)
        plt.show()
        """

        for cam_idx in range(len(cameras)):
            # Check whether the board is in front of the the image
            pt_3d = cameras[cam_idx].R @ pts_board + cameras[cam_idx].t
            if not board.isProjectionValid(cam_points_2d[cam_idx]) or np.min(pt_3d[2]) < 1e-3:
                cam_points_2d[cam_idx] = -np.ones_like(cam_points_2d[cam_idx])
            elif cameras[cam_idx].noise_scale_img_diag is not None:
                cam_points_2d[cam_idx] += np.random.normal(0, cameras[cam_idx].img_diag * cameras[cam_idx].noise_scale_img_diag, cam_points_2d[cam_idx].shape)

        ### test
        pts_inside_camera = np.zeros(len(cameras), dtype=bool)
        for ii, pts_2d in enumerate(cam_points_2d):
            mask = insideImageMask(pts_2d, cameras[ii].img_width, cameras[ii].img_height)
            # cam_points_2d[ii] = cam_points_2d[ii][:,mask]
            pts_inside_camera[ii] = mask.all()
            # print(pts_inside, end=' ')
        # print('from max inside', pts_board.shape[1])
        ###

        if pts_inside_camera.sum() >= 2:
            valid_frames_per_camera += np.array(pts_inside_camera, int)
            print(valid_frames_per_camera)
            points_2d.append(np.stack(cam_points_2d))
            points_3d.append(pts_board)

            R_used.append(R_board)
            t_used.append(R_board @ (board.t_origin - points_board_mean) + points_board_mean + t_board)

            if len(points_2d) >= num_frames and (valid_frames_per_camera >= MIN_FRAMES_PER_CAM).all():
                print('tried samples', frame)
                break

    VIDEOS_FPS = 5
    VIDEOS_DPI = 250
    MAX_FRAMES = 100
    if save_proj_animation is not None: animation2D(board, cameras, points_2d, save_proj_animation, VIDEOS_FPS, VIDEOS_DPI, MAX_FRAMES)
    if save_3d_animation is not None: animation3D(board, cameras, points_3d, save_3d_animation, VIDEOS_FPS, VIDEOS_DPI, MAX_FRAMES)

    print('number of found frames', len(points_2d))
    return np.stack(points_2d), np.stack(points_3d), np.stack(R_used), np.stack(t_used)

def createConfigFile(fname, params):
    file = open(fname, 'w')

    def writeDict(dict_write, tab):
        for key, value in dict_write.items():
            if isinstance(value, dict):
                file.write(tab+key+' :\n')
                writeDict(value, tab+'  ')
            else:
                file.write(tab+key+' : '+str(value)+'\n')
        file.write('\n')
    writeDict(params, '')
    file.close()

def generateRoomConfiguration():
    params = {'NAME' : '"room_corners"', 'NUM_SAMPLES': 1, 'SEED': 0, 'MAX_FRAMES' : 50, 'MAX_RANDOM_ITERS' : 100000, 'NUM_CAMERAS': 4,
              'BOARD': {'WIDTH':9, 'HEIGHT':7, 'SQUARE_LEN':0.08, 'T_LIMIT': [[-0.2,0.2], [-0.2,0.2], [-0.1,0.1]], 'EULER_LIMIT': [[-45, 45], [-180, 180], [-45, 45]], 'T_ORIGIN': [-0.3,0,1.5]}}
    params['CAMERA1'] = {'FX': [1200, 1200], 'FY_DEVIATION': 'null', 'IMG_WIDTH': 1500, 'IMG_HEIGHT': 1080, 'EULER_LIMIT': 'null', 'T_LIMIT': 'null', 'NOISE_SCALE': 3.0e-4, 'FISHEYE': False, 'DIST': [[5.2e-1,5.2e-1], [0,0], [0,0], [0,0], [0,0]]}
    params['CAMERA2'] = {'FX': [1000, 1000], 'FY_DEVIATION': 'null', 'IMG_WIDTH': 1300, 'IMG_HEIGHT': 1000, 'EULER_LIMIT': [[0,0], [90,90], [0,0]], 'T_LIMIT': [[-2.0,-2.0], [0.0, 0.0], [1.5, 1.5]], 'NOISE_SCALE': 3.5e-4, 'FISHEYE': False, 'DIST': [[3.2e-1,3.2e-1], [0,0], [0,0], [0,0], [0,0]]}
    params['CAMERA3'] = {'FX': [1000, 1000], 'FY_DEVIATION': 'null', 'IMG_WIDTH': 1300, 'IMG_HEIGHT': 1000, 'EULER_LIMIT': [[0,0], [-90,-90], [0,0]], 'T_LIMIT': [[2.0,2.0], [0.0, 0.0], [1.5, 1.5]], 'NOISE_SCALE': 4.0e-4, 'FISHEYE': False, 'DIST': [[6.2e-1,6.2e-1], [0,0], [0,0], [0,0], [0,0]]}
    params['CAMERA4'] = {'FX': [1000, 1000], 'FY_DEVIATION': 'null', 'IMG_WIDTH': 1300, 'IMG_HEIGHT': 1000, 'EULER_LIMIT': [[0,0], [180,180], [0,0]], 'T_LIMIT': [[0.0,0.0], [0.0, 0.0], [3.0, 3.0]], 'NOISE_SCALE': 3.2e-4, 'FISHEYE': False, 'DIST': [[4.2e-1,4.2e-1], [0,0], [0,0], [0,0], [0,0]]}
    createConfigFile('python/configs/config_room_corners.yaml', params)

def generateCircularCameras():
    rand_gen = RandGen(0)
    params = {'NAME' : '"circular"', 'NUM_SAMPLES': 1, 'SEED': 0, 'MAX_FRAMES' : 70, 'MAX_RANDOM_ITERS' : 100000, 'NUM_CAMERAS': 9,
        'BOARD': {'WIDTH': 9, 'HEIGHT': 7, 'SQUARE_LEN':0.08, 'T_LIMIT': [[-0.2,0.2], [-0.2,0.2], [-0.1,0.1]], 'EULER_LIMIT': [[-45, 45], [-180, 180], [-45, 45]], 'T_ORIGIN': [-0.3,0,2.2]}}

    dist = 1.1
    xs = np.arange(dist, dist*(params['NUM_CAMERAS']//4)+1e-3, dist)
    xs = np.concatenate((xs, xs[::-1]))
    xs = np.concatenate((xs, -xs))
    dist_z = 0.90
    zs = np.arange(dist_z, dist_z*(params['NUM_CAMERAS']//2)+1e-3, dist_z)
    zs = np.concatenate((zs, zs[::-1]))
    yaw = np.linspace(0, -360, params['NUM_CAMERAS']+1)[1:-1]
    for i in range(9):
        fx = rand_gen.randRange(900, 1300)
        d0 = rand_gen.randRange(4e-1, 7e-1)
        euler_limit = 'null'
        t_limit = 'null'
        if i > 0:
            euler_limit = [[0,0], [yaw[i-1], yaw[i-1]], [0,0]]
            t_limit = [[xs[i-1], xs[i-1]], [0,0], [zs[i-1], zs[i-1]]]
        params['CAMERA'+str((i+1))] = {'FX': [fx, fx], 'FY_DEVIATION': 'null', 'IMG_WIDTH': int(rand_gen.randRange(1200, 1600)), 'IMG_HEIGHT': int(rand_gen.randRange(800, 1200)),
            'EULER_LIMIT': euler_limit, 'T_LIMIT': t_limit, 'NOISE_SCALE': rand_gen.randRange(2e-4, 5e-4), 'FISHEYE': False, 'DIST': [[d0,d0], [0,0], [0,0], [0,0], [0,0]]}

    createConfigFile('python/configs/config_circular.yaml', params)

def getCamerasFromCfg(cfg):
    cameras = []
    for i in range(cfg['NUM_CAMERAS']):
        cameras.append(Camera(i, cfg['CAMERA' + str(i+1)]['IMG_WIDTH'], cfg['CAMERA' + str(i+1)]['IMG_HEIGHT'],
              cfg['CAMERA' + str(i+1)]['FX'], cfg['CAMERA' + str(i+1)]['EULER_LIMIT'], cfg['CAMERA' + str(i+1)]['T_LIMIT'],
              cfg['CAMERA' + str(i+1)]['FISHEYE'], cfg['CAMERA' + str(i+1)]['FY_DEVIATION'],
              noise_scale_img_diag=cfg['CAMERA' + str(i+1)]['NOISE_SCALE'], distortion_limit=cfg['CAMERA' + str(i+1)]['DIST']))
    return cameras

def main(cfg_name, save_folder):
    cfg = yaml.safe_load(open(cfg_name, 'r'))
    print(cfg)
    np.random.seed(cfg['SEED'])
    for trial in range(cfg['NUM_SAMPLES']):
        Path(save_folder).mkdir(exist_ok=True, parents=True)

        checkerboard = CheckerBoard(cfg['BOARD']['WIDTH'], cfg['BOARD']['HEIGHT'], cfg['BOARD']['SQUARE_LEN'], cfg['BOARD']['EULER_LIMIT'], cfg['BOARD']['T_LIMIT'], cfg['BOARD']['T_ORIGIN'])
        cameras = getCamerasFromCfg(cfg)
        points_2d, points_3d, R_used, t_used = generateAll(cameras, checkerboard, cfg['MAX_FRAMES'], RandGen(cfg['SEED']), cfg['MAX_RANDOM_ITERS'], save_folder+'plots_projections.mp4', save_folder+'board_cameras.mp4')

        for i in range(len(cameras)):
            print('Camera', i)
            print('K', cameras[i].K)
            print('R', cameras[i].R)
            print('t', cameras[i].t.flatten())
            print('distortion', cameras[i].distortion.flatten())
            print('-----------------------------')

        imgs_width_height = [[cam.img_width, cam.img_height] for cam in cameras]
        is_fisheye = [cam.is_fisheye for cam in cameras]
        export2JSON(checkerboard.pattern, points_2d, imgs_width_height, is_fisheye, save_folder+'opencv_sample_'+cfg['NAME']+'.json')
        saveKDRT(cameras, save_folder+'gt.txt')

        file = open(save_folder + "gt.txt", "a")
        for i in range(R_used.shape[0]):
            writeMatrix(file, 'R_%d' % i, R_used[i])
            writeMatrix(file, 'T_%d' % i, t_used[i])

        poses = dict()
        for idx in range(len(R_used)):
            poses['frame_%d' % idx] = {'R': R_used[idx].tolist(), 'T': t_used[idx].tolist()}

        with open(os.path.join(save_folder, "gt_poses.json"), 'wt') as gt:
            gt.write(json.dumps(poses, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config file, e.g., config_cv_test.yaml')
    parser.add_argument('--output_folder', type=str, default='', help='output folder')
    params, _ = parser.parse_known_args()
    main(params.cfg, params.output_folder)
