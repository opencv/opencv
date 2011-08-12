#!/usr/bin/python
import urllib2
import cv2.cv as cv
from random import randint
MAX_CLUSTERS = 5

if __name__ == "__main__":

    color_tab = [
        cv.CV_RGB(255, 0,0),
        cv.CV_RGB(0, 255, 0),
        cv.CV_RGB(100, 100, 255),
        cv.CV_RGB(255, 0,255),
        cv.CV_RGB(255, 255, 0)]
    img = cv.CreateImage((500, 500), 8, 3)
    rng = cv.RNG(-1)

    cv.NamedWindow("clusters", 1)
        
    while True:
        cluster_count = randint(2, MAX_CLUSTERS)
        sample_count = randint(1, 1000)
        points = cv.CreateMat(sample_count, 1, cv.CV_32FC2)
        clusters = cv.CreateMat(sample_count, 1, cv.CV_32SC1)
        
        # generate random sample from multigaussian distribution
        for k in range(cluster_count):
            center = (cv.RandInt(rng)%img.width, cv.RandInt(rng)%img.height)
            first = k*sample_count/cluster_count
            last = sample_count
            if k != cluster_count:
                last = (k+1)*sample_count/cluster_count

            point_chunk = cv.GetRows(points, first, last)
                        
            cv.RandArr(rng, point_chunk, cv.CV_RAND_NORMAL,
                       cv.Scalar(center[0], center[1], 0, 0),
                       cv.Scalar(img.width*0.1, img.height*0.1, 0, 0))
        

        # shuffle samples 
        cv.RandShuffle(points, rng)

        cv.KMeans2(points, cluster_count, clusters,
                   (cv.CV_TERMCRIT_EPS + cv.CV_TERMCRIT_ITER, 10, 1.0))

        cv.Zero(img)

        for i in range(sample_count):
            cluster_idx = int(clusters[i, 0])
            pt = (cv.Round(points[i, 0][0]), cv.Round(points[i, 0][1]))
            cv.Circle(img, pt, 2, color_tab[cluster_idx], cv.CV_FILLED, cv.CV_AA, 0)

        cv.ShowImage("clusters", img)

        key = cv.WaitKey(0) % 0x100
        if key in [27, ord('q'), ord('Q')]:
            break
    
    cv.DestroyWindow("clusters")
