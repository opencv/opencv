import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import os

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

testDataPath = os.getenv("OPENCV_TEST_DATA_PATH", default=None)
srcDepth = np.asarray(Image.open(testDataPath + "/cv/rgbd/depth.png"))
srcRgb   = np.asarray(Image.open(testDataPath + "/cv/rgbd/rgb.png"))

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

def reprojectRtProject(image, K, depthFactor, rmat, tmat):
    Kinv = np.linalg.inv(K)
    xsz, ysz = image.shape[1], image.shape[0]
    projected = np.zeros((ysz, xsz, 3))
    for y in range(ysz):
        for x in range(xsz):
            z = image[y, x]/depthFactor

            v = K @ (rmat @ Kinv @ np.array([x*z, y*z, z]).T + tmat[:, 0])

            if z > 0:
                projected[y, x, :] = v[:]
    
    return projected

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
def splat(projected, maxv, rgb):
    xsz, ysz = projected.shape[1], projected.shape[0]
    depth = np.full((ysz, xsz), maxv, np.float32)
    colors = np.full((ysz, xsz, 3), 0, np.uint8)
    for y in range(ysz):
        for x in range(xsz):
            p = projected[y, x, :]
            z = p[2]
            if z > 0:
                u, v = int(p[0]/z), int(p[1]/z)
                okuv = (u >= 0 and v >= 0 and u < xsz and v < ysz)
                if okuv and depth[v, u] > z:
                    depth[v, u] = z
                    colors[v, u, :] = rgb[y, x, :]
    return depth, colors

maxv = depthFactor
dstDepth, dstRgb = splat(reprojectRtProject(srcDepth, K, depthFactor, rmat, tmat), maxv, srcRgb)
dstDepth[dstDepth >= maxv] = 0
dstDepth = (dstDepth*depthFactor).astype(np.uint16)

Image.fromarray(dstDepth).save(testDataPath + "/cv/rgbd/warpedDepth.png")
Image.fromarray(dstRgb  ).save(testDataPath + "/cv/rgbd/warpedRgb.png")

# debug
def outFile(path, ptsimg):
    f = open(path, "w")
    for y in range(ptsimg.shape[0]):
        for x in range(ptsimg.shape[1]):
            v = ptsimg[y, x, :]
            if v[2] > 0:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    f.close()

outObj = False
if outObj:
    objdir = "/path/to/objdir/"
    outFile(objdir + "reproj_rot_proj.obj", reproject(dstDepth, depthFactor, K))
    outFile(objdir + "rotated.obj", reprojectRt(srcDepth, K, depthFactor, rmat, tmat))

