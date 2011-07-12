import sys
import math
import time
import random

import numpy
import transformations
import cv2.cv as cv

def clamp(a, x, b):
    return numpy.maximum(a, numpy.minimum(x, b))

def norm(v):
    mag = numpy.sqrt(sum([e * e for e in v]))
    return v / mag

class Vec3:
    def __init__(self, x, y, z):
        self.v = (x, y, z)
    def x(self):
        return self.v[0]
    def y(self):
        return self.v[1]
    def z(self):
        return self.v[2]
    def __repr__(self):
        return "<Vec3 (%s,%s,%s)>" % tuple([repr(c) for c in self.v])
    def __add__(self, other):
        return Vec3(*[self.v[i] + other.v[i] for i in range(3)])
    def __sub__(self, other):
        return Vec3(*[self.v[i] - other.v[i] for i in range(3)])
    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(*[self.v[i] * other.v[i] for i in range(3)])
        else:
            return Vec3(*[self.v[i] * other for i in range(3)])
    def mag2(self):
        return sum([e * e for e in self.v])
    def __abs__(self):
        return numpy.sqrt(sum([e * e for e in self.v]))
    def norm(self):
        return self * (1.0 / abs(self))
    def dot(self, other):
        return sum([self.v[i] * other.v[i] for i in range(3)])
    def cross(self, other):
        (ax, ay, az) = self.v
        (bx, by, bz) = other.v
        return Vec3(ay * bz - by * az, az * bx - bz * ax, ax * by - bx * ay)


class Ray:

    def __init__(self, o, d):
        self.o = o
        self.d = d

    def project(self, d):
        return self.o + self.d * d

class Camera:

    def __init__(self, F):
        R = Vec3(1., 0., 0.)
        U = Vec3(0, 1., 0)
        self.center = Vec3(0, 0, 0)
        self.pcenter = Vec3(0, 0, F)
        self.up = U
        self.right = R

    def genray(self, x, y):
        """ -1 <= y <= 1 """
        r = numpy.sqrt(x * x + y * y)
        if 0:
            rprime = r + (0.17 * r**2)
        else:
            rprime = (10 * numpy.sqrt(17 * r + 25) - 50) / 17
        print "scale", rprime / r
        x *= rprime / r
        y *= rprime / r
        o = self.center
        r = (self.pcenter + (self.right * x) + (self.up * y)) - o
        return Ray(o, r.norm())

class Sphere:

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def hit(self, r):
        # a = mag2(r.d)
        a = 1.
        v = r.o - self.center
        b = 2 * r.d.dot(v)
        c = self.center.mag2() + r.o.mag2() + -2 * self.center.dot(r.o) - (self.radius ** 2)
        det = (b * b) - (4 * c)
        pred = 0 < det

        sq = numpy.sqrt(abs(det))
        h0 = (-b - sq) / (2)
        h1 = (-b + sq) / (2)

        h = numpy.minimum(h0, h1)

        pred = pred & (h > 0)
        normal = (r.project(h) - self.center) * (1.0 / self.radius)
        return (pred, numpy.where(pred, h, 999999.), normal)

def pt2plane(p, plane):
    return p.dot(plane) * (1. / abs(plane))

class Plane:

    def __init__(self, p, n, right):
        self.D = -pt2plane(p, n)
        self.Pn = n
        self.right = right
        self.rightD = -pt2plane(p, right)
        self.up = n.cross(right)
        self.upD = -pt2plane(p, self.up)

    def hit(self, r):
        Vd = self.Pn.dot(r.d)
        V0 = -(self.Pn.dot(r.o) + self.D)
        h = V0 / Vd
        pred = (0 <= h)

        return (pred, numpy.where(pred, h, 999999.), self.Pn)

    def localxy(self, loc):
        x = (loc.dot(self.right) + self.rightD)
        y = (loc.dot(self.up) + self.upD)
        return (x, y)

# lena = numpy.fromstring(cv.LoadImage("../samples/c/lena.jpg", 0).tostring(), numpy.uint8) / 255.0

def texture(xy):
    x,y = xy
    xa = numpy.floor(x * 512)
    ya = numpy.floor(y * 512)
    a = (512 * ya) + xa
    safe = (0 <= x) & (0 <= y) & (x < 1) & (y < 1)
    if 0:
        a = numpy.where(safe, a, 0).astype(numpy.int)
        return numpy.where(safe, numpy.take(lena, a), 0.0)
    else:
        xi = numpy.floor(x * 11).astype(numpy.int)
        yi = numpy.floor(y * 11).astype(numpy.int)
        inside = (1 <= xi) & (xi < 10) & (2 <= yi) & (yi < 9)
        checker = (xi & 1) ^ (yi & 1)
        final = numpy.where(inside, checker, 1.0)
        return numpy.where(safe, final, 0.5)

def under(vv, m):
    return Vec3(*(numpy.dot(m, vv.v + (1,))[:3]))

class Renderer:

    def __init__(self, w, h, oversample):
        self.w = w
        self.h = h

        random.seed(1)
        x = numpy.arange(self.w*self.h) % self.w
        y = numpy.floor(numpy.arange(self.w*self.h) / self.w)
        h2 = h / 2.0
        w2 = w / 2.0
        self.r = [ None ] * oversample
        for o in range(oversample):
            stoch_x = numpy.random.rand(self.w * self.h)
            stoch_y = numpy.random.rand(self.w * self.h)
            nx = (x + stoch_x - 0.5 - w2) / h2
            ny = (y + stoch_y - 0.5 - h2) / h2
            self.r[o] = cam.genray(nx, ny)

        self.rnds = [random.random() for i in range(10)]

    def frame(self, i):

        rnds = self.rnds
        roll = math.sin(i * .01 * rnds[0] + rnds[1])
        pitch = math.sin(i * .01 * rnds[2] + rnds[3])
        yaw = math.pi * math.sin(i * .01 * rnds[4] + rnds[5])
        x = math.sin(i * 0.01 * rnds[6])
        y = math.sin(i * 0.01 * rnds[7])

        x,y,z = -0.5,0.5,1
        roll,pitch,yaw = (0,0,0)

        z = 4 + 3 * math.sin(i * 0.1 * rnds[8])
        print z

        rz = transformations.euler_matrix(roll, pitch, yaw)
        p = Plane(Vec3(x, y, z), under(Vec3(0,0,-1), rz), under(Vec3(1, 0, 0), rz))

        acc = 0
        for r in self.r:
            (pred, h, norm) = p.hit(r)
            l = numpy.where(pred, texture(p.localxy(r.project(h))), 0.0)
            acc += l
        acc *= (1.0 / len(self.r))

        # print "took", time.time() - st

        img = cv.CreateMat(self.h, self.w, cv.CV_8UC1)
        cv.SetData(img, (clamp(0, acc, 1) * 255).astype(numpy.uint8).tostring(), self.w)
        return img

#########################################################################

num_x_ints = 8
num_y_ints = 6
num_pts = num_x_ints * num_y_ints

def get_corners(mono, refine = False):
    (ok, corners) = cv.FindChessboardCorners(mono, (num_x_ints, num_y_ints), cv.CV_CALIB_CB_ADAPTIVE_THRESH | cv.CV_CALIB_CB_NORMALIZE_IMAGE)
    if refine and ok:
        corners = cv.FindCornerSubPix(mono, corners, (5,5), (-1,-1), ( cv.CV_TERMCRIT_EPS+cv.CV_TERMCRIT_ITER, 30, 0.1 ))
    return (ok, corners)

def mk_object_points(nimages, squaresize = 1):
    opts = cv.CreateMat(nimages * num_pts, 3, cv.CV_32FC1)
    for i in range(nimages):
        for j in range(num_pts):
            opts[i * num_pts + j, 0] = (j / num_x_ints) * squaresize
            opts[i * num_pts + j, 1] = (j % num_x_ints) * squaresize
            opts[i * num_pts + j, 2] = 0
    return opts

def mk_image_points(goodcorners):
    ipts = cv.CreateMat(len(goodcorners) * num_pts, 2, cv.CV_32FC1)
    for (i, co) in enumerate(goodcorners):
        for j in range(num_pts):
            ipts[i * num_pts + j, 0] = co[j][0]
            ipts[i * num_pts + j, 1] = co[j][1]
    return ipts

def mk_point_counts(nimages):
    npts = cv.CreateMat(nimages, 1, cv.CV_32SC1)
    for i in range(nimages):
        npts[i, 0] = num_pts
    return npts

def cvmat_iterator(cvmat):
    for i in range(cvmat.rows):
        for j in range(cvmat.cols):
            yield cvmat[i,j]

cam = Camera(3.0)
rend = Renderer(640, 480, 2)
cv.NamedWindow("snap")

#images = [rend.frame(i) for i in range(0, 2000, 400)]
images = [rend.frame(i) for i in [1200]]

if 0:
    for i,img in enumerate(images):
        cv.SaveImage("final/%06d.png" % i, img)

size = cv.GetSize(images[0])
corners = [get_corners(i) for i in images]

goodcorners = [co for (im, (ok, co)) in zip(images, corners) if ok]

def checkerboard_error(xformed):
    def pt2line(a, b, c):
        x0,y0 = a
        x1,y1 = b
        x2,y2 = c
        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    errorsum = 0.
    for im in xformed:
        for row in range(6):
            l0 = im[8 * row]
            l1 = im[8 * row + 7]
            for col in range(1, 7):
                e = pt2line(im[8 * row + col], l0, l1)
                #print "row", row, "e", e
                errorsum += e

    return errorsum

if True:
    from scipy.optimize import fmin

    def xf(pt, poly):
        x, y = pt
        r = math.sqrt((x - 320) ** 2 + (y - 240) ** 2)
        fr = poly(r) / r
        return (320 + (x - 320) * fr, 240 + (y - 240) * fr)
    def silly(p, goodcorners):
    #    print "eval", p

        d = 1.0 # - sum(p)
        poly = numpy.poly1d(list(p) + [d, 0.])

        xformed = [[xf(pt, poly) for pt in co] for co in goodcorners]

        return checkerboard_error(xformed)

    x0 = [ 0. ]
    #print silly(x0, goodcorners)
    print "initial error", silly(x0, goodcorners)
    xopt = fmin(silly, x0, args=(goodcorners,))
    print "xopt", xopt
    print "final error", silly(xopt, goodcorners)

    d = 1.0 # - sum(xopt)
    poly = numpy.poly1d(list(xopt) + [d, 0.])
    print "final polynomial"
    print poly

    for co in goodcorners:
        scrib = cv.CreateMat(480, 640, cv.CV_8UC3)
        cv.SetZero(scrib)
        cv.DrawChessboardCorners(scrib, (num_x_ints, num_y_ints), [xf(pt, poly) for pt in co], True)
        cv.ShowImage("snap", scrib)
        cv.WaitKey()

    sys.exit(0)

for (i, (img, (ok, co))) in enumerate(zip(images, corners)):
    scrib = cv.CreateMat(img.rows, img.cols, cv.CV_8UC3)
    cv.CvtColor(img, scrib, cv.CV_GRAY2BGR)
    if ok:
        cv.DrawChessboardCorners(scrib, (num_x_ints, num_y_ints), co, True)
    cv.ShowImage("snap", scrib)
    cv.WaitKey()

print len(goodcorners)
ipts = mk_image_points(goodcorners)
opts = mk_object_points(len(goodcorners), .1)
npts = mk_point_counts(len(goodcorners))

intrinsics = cv.CreateMat(3, 3, cv.CV_64FC1)
distortion = cv.CreateMat(4, 1, cv.CV_64FC1)
cv.SetZero(intrinsics)
cv.SetZero(distortion)
# focal lengths have 1/1 ratio
intrinsics[0,0] = 1.0
intrinsics[1,1] = 1.0
cv.CalibrateCamera2(opts, ipts, npts,
           cv.GetSize(images[0]),
           intrinsics,
           distortion,
           cv.CreateMat(len(goodcorners), 3, cv.CV_32FC1),
           cv.CreateMat(len(goodcorners), 3, cv.CV_32FC1),
           flags = 0) # cv.CV_CALIB_ZERO_TANGENT_DIST)
print "D =", list(cvmat_iterator(distortion))
print "K =", list(cvmat_iterator(intrinsics))
mapx = cv.CreateImage((640, 480), cv.IPL_DEPTH_32F, 1)
mapy = cv.CreateImage((640, 480), cv.IPL_DEPTH_32F, 1)
cv.InitUndistortMap(intrinsics, distortion, mapx, mapy)
for img in images:
    r = cv.CloneMat(img)
    cv.Remap(img, r, mapx, mapy)
    cv.ShowImage("snap", r)
    cv.WaitKey()
