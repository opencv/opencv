import numpy as np
from scipy.spatial.transform import Rotation
import imageio
# optional, works slower w/o it
from numba import jit

depthFactor = 5000
psize = (640, 480)
fx = 525.0
fy = 525.0
cx = psize[0]/2-0.5
cy = psize[1]/2-0.5
K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]])
# some random transform
rmat = Rotation.from_rotvec(np.array([0.1, 0.2, 0.3])).as_dcm()
tmat = np.array([[-0.04, 0.05, 0.6]]).T
rtmat = np.vstack((np.hstack((rmat, tmat)), np.array([[0, 0, 0, 1]])))

#TODO: warp rgb image as well
testDataPath = "/path/to/sources/opencv_extra/testdata"
srcDepth = imageio.imread(testDataPath + "/cv/rgbd/depth.png")

@jit
def reproject(image, df, K):
    Kinv = np.linalg.inv(K)
    xsz, ysz = image.shape[1], image.shape[0]
    reprojected = np.zeros((ysz, xsz, 3))
    for y in range(ysz):
        for x in range(xsz):
            z = image[y, x]/df

            v = Kinv @ np.array([x*z, y*z, z]).T

            #xv = (x - cx)/fx * z
            #yv = (y - cy)/fy * z
            #zv = z

            reprojected[y, x, :] = v[:]
    return reprojected

@jit
def reprojectRtProject(image, K, depthFactor, rmat, tmat):
    Kinv = np.linalg.inv(K)
    xsz, ysz = image.shape[1], image.shape[0]
    projected = np.zeros((ysz, xsz, 3))
    for y in range(ysz):
        for x in range(xsz):
            z = image[y, x]/depthFactor

            v = K @ (rmat @ Kinv @ np.array([x*z, y*z, z]).T + tmat[:, 0])

            projected[y, x, :] = v[:]
    
    return projected

@jit
def reprojectRt(image, K, depthFactor, rmat, tmat):
    Kinv = np.linalg.inv(K)
    xsz, ysz = image.shape[1], image.shape[0]
    rotated = np.zeros((ysz, xsz, 3))
    for y in range(ysz):
        for x in range(xsz):
            z = image[y, x]/depthFactor

            v = rmat @ Kinv @ np.array([x*z, y*z, z]).T + tmat[:, 0]

            rotated[y, x, :] = v[:]
    
    return rotated

# put projected points on a depth map
@jit
def splat(projected, maxv):
    xsz, ysz = projected.shape[1], projected.shape[0]
    depth = np.full((ysz, xsz), maxv, np.float32)
    for y in range(ysz):
        for x in range(xsz):
            p = projected[y, x, :]
            z = p[2]
            if z > 0:
                u, v = int(p[0]/z), int(p[1]/z)
                okuv = (u >= 0 and v >= 0 and u < xsz and v < ysz)
                if okuv and depth[v, u] > z:
                    depth[v, u] = z
    return depth

maxv = depthFactor
im2 = splat(reprojectRtProject(srcDepth, K, depthFactor, rmat, tmat), maxv)
im2[im2 >= maxv] = 0
im2 = im2*depthFactor

imageio.imwrite(testDataPath + "/cv/rgbd/warped.png", im2)

# debug

outObj = False
def outFile(path, ptsimg):
    f = open(path, "w")
    for y in range(ptsimg.shape[0]):
        for x in range(ptsimg.shape[1]):
            v = ptsimg[y, x, :]
            if v[2] > 0:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    f.close()

if outObj:
    objdir = "/path/to/objdir/"
    outFile(objdir + "reproj_rot_proj.obj", reproject(im2, depthFactor, K))
    outFile(objdir + "rotated.obj", reprojectRt(srcDepth, K, depthFactor, rmat, tmat))


