import cv
import time
from pydmtx import DataMatrix
import numpy
import sys
import math

def absnorm8(im, im8):
    """ im may be any single-channel image type.  Return an 8-bit version, absolute value, normalized so that max is 255 """
    (minVal, maxVal, _, _) = cv.MinMaxLoc(im)
    cv.ConvertScaleAbs(im, im8, 255 / max(abs(minVal), abs(maxVal)), 0)
    return im8

font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, thickness = 2, lineType = cv.CV_AA)
if 0:
    started = time.time()
    print dm_write.decode(bg.width, bg.height, buffer(bg.tostring()), max_count = 1, min_edge = 12, max_edge = 13, shape = DataMatrix.DmtxSymbol10x10) # , timeout = 10)
    print "took", time.time() - started

class DmtxFinder:
    def __init__(self):
        self.cache = {}
        self.dm = DataMatrix()

    def Cached(self, name, rows, cols, type):
        key = (name, rows, cols)
        if not key in self.cache:
            self.cache[key] = cv.CreateMat(rows, cols, type)
        return self.cache[key]

    def find0(self, img):
        started = time.time()
        self.dm.decode(img.width,
                       img.height,
                       buffer(img.tostring()),
                       max_count = 4,
                       #min_edge = 6,
                       #max_edge = 19      # Units of 2 pixels
                       )
        print "brute", time.time() - started
        found = {}
        for i in range(self.dm.count()):
            stats = dm_read.stats(i + 1)
            print stats
            found[stats[0]] = stats[1]
        return found

    def find(self, img):
        started = time.time()
        gray = self.Cached('gray', img.height, img.width, cv.CV_8UC1)
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

        sobel = self.Cached('sobel', img.height, img.width, cv.CV_16SC1)
        sobely = self.Cached('sobely', img.height, img.width, cv.CV_16SC1)

        cv.Sobel(gray, sobel, 1, 0)
        cv.Sobel(gray, sobely, 0, 1)
        cv.Add(sobel, sobely, sobel)

        sobel8 = self.Cached('sobel8', sobel.height, sobel.width, cv.CV_8UC1)
        absnorm8(sobel, sobel8)
        cv.Threshold(sobel8, sobel8, 128.0, 255.0, cv.CV_THRESH_BINARY)

        sobel_integral = self.Cached('sobel_integral', img.height + 1, img.width + 1, cv.CV_32SC1)
        cv.Integral(sobel8, sobel_integral)

        d = 16
        _x1y1 = cv.GetSubRect(sobel_integral, (0, 0, sobel_integral.cols - d, sobel_integral.rows - d))
        _x1y2 = cv.GetSubRect(sobel_integral, (0, d, sobel_integral.cols - d, sobel_integral.rows - d))
        _x2y1 = cv.GetSubRect(sobel_integral, (d, 0, sobel_integral.cols - d, sobel_integral.rows - d))
        _x2y2 = cv.GetSubRect(sobel_integral, (d, d, sobel_integral.cols - d, sobel_integral.rows - d))

        summation = cv.CloneMat(_x2y2)
        cv.Sub(summation, _x1y2, summation)
        cv.Sub(summation, _x2y1, summation)
        cv.Add(summation, _x1y1, summation)
        sum8 = self.Cached('sum8', summation.height, summation.width, cv.CV_8UC1)
        absnorm8(summation, sum8)
        cv.Threshold(sum8, sum8, 32.0, 255.0, cv.CV_THRESH_BINARY)

        cv.ShowImage("sum8", sum8)
        seq = cv.FindContours(sum8, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL)
        subimg = cv.GetSubRect(img, (d / 2, d / 2, sum8.cols, sum8.rows))
        t_cull = time.time() - started

        seqs = []
        while seq:
            seqs.append(seq)
            seq = seq.h_next()

        started = time.time()
        found = {}
        print 'seqs', len(seqs)
        for seq in seqs:
            area = cv.ContourArea(seq)
            if area > 1000:
                rect = cv.BoundingRect(seq)
                edge = int((14 / 14.) * math.sqrt(area) / 2 + 0.5)
                candidate = cv.GetSubRect(subimg, rect)
                sym = self.dm.decode(candidate.width,
                                     candidate.height,
                                     buffer(candidate.tostring()),
                                     max_count = 1,
                                     #min_edge = 6,
                                     #max_edge = int(edge)      # Units of 2 pixels
                                     )
                if sym:
                    onscreen = [(d / 2 + rect[0] + x, d / 2 + rect[1] + y) for (x, y) in self.dm.stats(1)[1]]
                    found[sym] = onscreen
                else:
                    print "FAILED"
        t_brute = time.time() - started
        print "cull took", t_cull, "brute", t_brute
        return found

bg = cv.CreateMat(1024, 1024, cv.CV_8UC3)
cv.Set(bg, cv.RGB(0, 0, 0))
df = DmtxFinder()

cv.NamedWindow("camera", 1)

def mkdmtx(msg):
    dm_write = DataMatrix()
    dm_write.encode(msg)
    pi = dm_write.image # .resize((14, 14))
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring())
    return cv_im

# test = [('WIL', (100,100))]: # , ('LOW', (250,100)), ('GAR', (300, 300)), ('AGE', (500, 300))]:

test = []
y = 10
for j in range(7):
    r = 28 + j * 4
    mr = r * math.sqrt(2)
    y += mr * 1.8
    test += [(str(deg) + "abcdefgh"[j], (50 + deg * 11, y), math.pi * deg / 180, r) for deg in range(0, 90, 10)]

for (msg, (x, y), angle, r) in test:
    map = cv.CreateMat(2, 3, cv.CV_32FC1)
    corners = [(x + r * math.cos(angle + th), y + r * math.sin(angle + th)) for th in [0, math.pi / 2, math.pi, 3 * math.pi / 4]]
    src = mkdmtx(msg)
    (sx, sy) = cv.GetSize(src)
    cv.GetAffineTransform([(0,0), (sx, 0), (sx, sy)], corners[:3], map)
    temp = cv.CreateMat(bg.rows, bg.cols, cv.CV_8UC3)
    cv.Set(temp, cv.RGB(0, 0, 0))
    cv.WarpAffine(src, temp, map)
    cv.Or(temp, bg, bg)


cv.ShowImage("comp", bg)
scribble = cv.CloneMat(bg)

if 0:
    for i in range(10):
        df.find(bg)

for (sym, coords) in df.find(bg).items():
    print sym
    cv.PolyLine(scribble, [coords], 1, cv.CV_RGB(255, 0,0), 1, lineType = cv.CV_AA)
    Xs = [x for (x, y) in coords]
    Ys = [y for (x, y) in coords]
    where = ((min(Xs) + max(Xs)) / 2, max(Ys) - 50)
    cv.PutText(scribble, sym, where, font, cv.RGB(0,255, 0))

cv.ShowImage("results", scribble)
cv.WaitKey()

sys.exit(0)

capture = cv.CaptureFromCAM(0)
while True:
    img = cv.QueryFrame(capture)
    cv.ShowImage("capture", img)
    print df.find(img)
    cv.WaitKey(6)
