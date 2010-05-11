"""
Find Squares in image by finding countours and filtering
"""
#Results slightly different from C version on same images, but is 
#otherwise ok

import math
import cv

def angle(pt1, pt2, pt0):
    "calculate angle contained by 3 points(x, y)"
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]

    nom = dx1*dx2 + dy1*dy2
    denom = math.sqrt( (dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2) + 1e-10 )
    ang = nom / denom
    return ang

def is_square(contour):
    """
    Squareness checker

    Square contours should:
        -have 4 vertices after approximation, 
        -have relatively large area (to filter out noisy contours)
        -be convex.
        -have angles between sides close to 90deg (cos(ang) ~0 )
    Note: absolute value of an area is used because area may be
    positive or negative - in accordance with the contour orientation
    """

    area = math.fabs( cv.ContourArea(contour) )
    isconvex = cv.CheckContourConvexity(contour)
    s = 0
    if len(contour) == 4 and area > 1000 and isconvex:
        for i in range(1, 4):
            # find minimum angle between joint edges (maximum of cosine)
            pt1 = contour[i]
            pt2 = contour[i-1]
            pt0 = contour[i-2]

            t = math.fabs(angle(pt0, pt1, pt2))
            if s <= t:s = t

        # if cosines of all angles are small (all angles are ~90 degree) 
        # then its a square
        if s < 0.3:return True

    return False       

def find_squares_from_binary( gray ):
    """
    use contour search to find squares in binary image
    returns list of numpy arrays containing 4 points
    """
    squares = []
    storage = cv.CreateMemStorage(0)
    contours = cv.FindContours(gray, storage, cv.CV_RETR_TREE, cv.CV_CHAIN_APPROX_SIMPLE, (0,0))  
    storage = cv.CreateMemStorage(0)
    while contours:
        #approximate contour with accuracy proportional to the contour perimeter
        arclength = cv.ArcLength(contours)
        polygon = cv.ApproxPoly( contours, storage, cv.CV_POLY_APPROX_DP, arclength * 0.02, 0)
        if is_square(polygon):
            squares.append(polygon[0:4])
        contours = contours.h_next()

    return squares

def find_squares4(color_img):
    """
    Finds multiple squares in image

    Steps:
    -Use Canny edge to highlight contours, and dilation to connect
    the edge segments.
    -Threshold the result to binary edge tokens
    -Use cv.FindContours: returns a cv.CvSequence of cv.CvContours
    -Filter each candidate: use Approx poly, keep only contours with 4 vertices, 
    enough area, and ~90deg angles.

    Return all squares contours in one flat list of arrays, 4 x,y points each.
    """
    #select even sizes only
    width, height = (color_img.width & -2, color_img.height & -2 )
    timg = cv.CloneImage( color_img ) # make a copy of input image
    gray = cv.CreateImage( (width,height), 8, 1 )

    # select the maximum ROI in the image
    cv.SetImageROI( timg, (0, 0, width, height) )

    # down-scale and upscale the image to filter out the noise
    pyr = cv.CreateImage( (width/2, height/2), 8, 3 )
    cv.PyrDown( timg, pyr, 7 )
    cv.PyrUp( pyr, timg, 7 )

    tgray = cv.CreateImage( (width,height), 8, 1 )
    squares = []

    # Find squares in every color plane of the image
    # Two methods, we use both:
    # 1. Canny to catch squares with gradient shading. Use upper threshold
    # from slider, set the lower to 0 (which forces edges merging). Then
    # dilate canny output to remove potential holes between edge segments.
    # 2. Binary thresholding at multiple levels
    N = 11
    for c in [0, 1, 2]:
        #extract the c-th color plane
        cv.SetImageCOI( timg, c+1 );
        cv.Copy( timg, tgray, None );
        cv.Canny( tgray, gray, 0, 50, 5 )
        cv.Dilate( gray, gray)
        squares = squares + find_squares_from_binary( gray )

        # Look for more squares at several threshold levels
        for l in range(1, N):
            cv.Threshold( tgray, gray, (l+1)*255/N, 255, cv.CV_THRESH_BINARY )
            squares = squares + find_squares_from_binary( gray )

    return squares


RED = (0,0,255)
GREEN = (0,255,0)
def draw_squares( color_img, squares ):
    """
    Squares is py list containing 4-pt numpy arrays. Step through the list
    and draw a polygon for each 4-group
    """
    color, othercolor = RED, GREEN
    for square in squares:
        cv.PolyLine(color_img, [square], True, color, 3, cv.CV_AA, 0)
        color, othercolor = othercolor, color

    cv.ShowImage(WNDNAME, color_img)

 
WNDNAME = "Squares Demo"
def main():
    """Open test color images, create display window, start the search"""
    cv.NamedWindow(WNDNAME, 1)
    for name in [ "../c/pic%d.png" % i for i in [1, 2, 3, 4, 5, 6] ]:
        img0 = cv.LoadImage(name, 1)
        try:
            img0
        except ValueError:
            print "Couldn't load %s\n" % name
            continue

        # slider deleted from C version, same here and use fixed Canny param=50
        img = cv.CloneImage(img0)

        cv.ShowImage(WNDNAME, img)

        # force the image processing
        draw_squares( img, find_squares4( img ) )
        
        # wait for key.
        if cv.WaitKey(-1) % 0x100 == 27:
            break

if __name__ == "__main__":
    main()    
