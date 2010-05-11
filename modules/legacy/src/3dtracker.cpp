/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2002, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#if _MSC_VER >= 1200
#pragma warning(disable:4786) // Disable MSVC warnings in the standard library.
#pragma warning(disable:4100)
#pragma warning(disable:4512)
#endif
#include <stdio.h>
#include <map>
#include <algorithm>
#if _MSC_VER >= 1200
#pragma warning(default:4100)
#pragma warning(default:4512)
#endif

#define ARRAY_SIZEOF(a) (sizeof(a)/sizeof((a)[0]))

static void FillObjectPoints(CvPoint3D32f *obj_points, CvSize etalon_size, float square_size);
static void DrawEtalon(IplImage *img, CvPoint2D32f *corners,
                       int corner_count, CvSize etalon_size, int draw_ordered);
static void MultMatrix(float rm[4][4], const float m1[4][4], const float m2[4][4]);
static void MultVectorMatrix(float rv[4], const float v[4], const float m[4][4]);
static CvPoint3D32f ImageCStoWorldCS(const Cv3dTrackerCameraInfo &camera_info, CvPoint2D32f p);
static bool intersection(CvPoint3D32f o1, CvPoint3D32f p1,
                         CvPoint3D32f o2, CvPoint3D32f p2,
                         CvPoint3D32f &r1, CvPoint3D32f &r2);

/////////////////////////////////
// cv3dTrackerCalibrateCameras //
/////////////////////////////////
CV_IMPL CvBool cv3dTrackerCalibrateCameras(int num_cameras,
                   const Cv3dTrackerCameraIntrinsics camera_intrinsics[], // size is num_cameras
                   CvSize etalon_size,
                   float square_size,
                   IplImage *samples[],                                   // size is num_cameras
                   Cv3dTrackerCameraInfo camera_info[])                   // size is num_cameras
{
    CV_FUNCNAME("cv3dTrackerCalibrateCameras");
    const int num_points = etalon_size.width * etalon_size.height;
    int cameras_done = 0;        // the number of cameras whose positions have been determined
    CvPoint3D32f *object_points = NULL; // real-world coordinates of checkerboard points
    CvPoint2D32f *points = NULL; // 2d coordinates of checkerboard points as seen by a camera
    IplImage *gray_img = NULL;   // temporary image for color conversion
    IplImage *tmp_img = NULL;    // temporary image used by FindChessboardCornerGuesses
    int c, i, j;

    if (etalon_size.width < 3 || etalon_size.height < 3)
        CV_ERROR(CV_StsBadArg, "Chess board size is invalid");

    for (c = 0; c < num_cameras; c++)
    {
        // CV_CHECK_IMAGE is not available in the cvaux library
        // so perform the checks inline.

        //CV_CALL(CV_CHECK_IMAGE(samples[c]));

        if( samples[c] == NULL )
            CV_ERROR( CV_HeaderIsNull, "Null image" );

        if( samples[c]->dataOrder != IPL_DATA_ORDER_PIXEL && samples[c]->nChannels > 1 )
            CV_ERROR( CV_BadOrder, "Unsupported image format" );

        if( samples[c]->maskROI != 0 || samples[c]->tileInfo != 0 )
            CV_ERROR( CV_StsBadArg, "Unsupported image format" );

        if( samples[c]->imageData == 0 )
            CV_ERROR( CV_BadDataPtr, "Null image data" );

        if( samples[c]->roi &&
            ((samples[c]->roi->xOffset | samples[c]->roi->yOffset
              | samples[c]->roi->width | samples[c]->roi->height) < 0 ||
             samples[c]->roi->xOffset + samples[c]->roi->width > samples[c]->width ||
             samples[c]->roi->yOffset + samples[c]->roi->height > samples[c]->height ||
             (unsigned) (samples[c]->roi->coi) > (unsigned) (samples[c]->nChannels)))
            CV_ERROR( CV_BadROISize, "Invalid ROI" );

        // End of CV_CHECK_IMAGE inline expansion

        if (samples[c]->depth != IPL_DEPTH_8U)
            CV_ERROR(CV_BadDepth, "Channel depth of source image must be 8");

        if (samples[c]->nChannels != 3 && samples[c]->nChannels != 1)
            CV_ERROR(CV_BadNumChannels, "Source image must have 1 or 3 channels");
    }

    CV_CALL(object_points = (CvPoint3D32f *)cvAlloc(num_points * sizeof(CvPoint3D32f)));
    CV_CALL(points = (CvPoint2D32f *)cvAlloc(num_points * sizeof(CvPoint2D32f)));

    // fill in the real-world coordinates of the checkerboard points
    FillObjectPoints(object_points, etalon_size, square_size);

    for (c = 0; c < num_cameras; c++)
    {
        CvSize image_size = cvSize(samples[c]->width, samples[c]->height);
        IplImage *img;

        // The input samples are not required to all have the same size or color
        // format. If they have different sizes, the temporary images are
        // reallocated as necessary.
        if (samples[c]->nChannels == 3)
        {
            // convert to gray
            if (gray_img == NULL || gray_img->width != samples[c]->width ||
                gray_img->height != samples[c]->height )
            {
                if (gray_img != NULL)
                    cvReleaseImage(&gray_img);
                CV_CALL(gray_img = cvCreateImage(image_size, IPL_DEPTH_8U, 1));
            }
            
            CV_CALL(cvCvtColor(samples[c], gray_img, CV_BGR2GRAY));

            img = gray_img;
        }
        else
        {
            // no color conversion required
            img = samples[c];
        }

        if (tmp_img == NULL || tmp_img->width != samples[c]->width ||
            tmp_img->height != samples[c]->height )
        {
            if (tmp_img != NULL)
                cvReleaseImage(&tmp_img);
            CV_CALL(tmp_img = cvCreateImage(image_size, IPL_DEPTH_8U, 1));
        }

        int count = num_points;
        bool found = cvFindChessBoardCornerGuesses(img, tmp_img, 0,
                                                   etalon_size, points, &count) != 0;
        if (count == 0)
            continue;
        
        // If found is true, it means all the points were found (count = num_points).
        // If found is false but count is non-zero, it means that not all points were found.

        cvFindCornerSubPix(img, points, count, cvSize(5,5), cvSize(-1,-1),
                    cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.01f));

        // If the image origin is BL (bottom-left), fix the y coordinates
        // so they are relative to the true top of the image.
        if (samples[c]->origin == IPL_ORIGIN_BL)
        {
            for (i = 0; i < count; i++)
                points[i].y = samples[c]->height - 1 - points[i].y;
        }

        if (found)
        {
            // Make sure x coordinates are increasing and y coordinates are decreasing.
            // (The y coordinate of point (0,0) should be the greatest, because the point
            // on the checkerboard that is the origin is nearest the bottom of the image.)
            // This is done after adjusting the y coordinates according to the image origin.
            if (points[0].x > points[1].x)
            {
                // reverse points in each row
                for (j = 0; j < etalon_size.height; j++)
                {
                    CvPoint2D32f *row = &points[j*etalon_size.width];
                    for (i = 0; i < etalon_size.width/2; i++)
                        std::swap(row[i], row[etalon_size.width-i-1]);
                }
            }

            if (points[0].y < points[etalon_size.width].y)
            {
                // reverse points in each column
                for (i = 0; i < etalon_size.width; i++)
                {
                    for (j = 0; j < etalon_size.height/2; j++)
                        std::swap(points[i+j*etalon_size.width],
                                  points[i+(etalon_size.height-j-1)*etalon_size.width]);
                }
            }
        }

        DrawEtalon(samples[c], points, count, etalon_size, found);

        if (!found)
            continue;

        float rotVect[3];
        float rotMatr[9];
        float transVect[3];

        cvFindExtrinsicCameraParams(count,
                                    image_size,
                                    points,
                                    object_points,
                                    const_cast<float *>(camera_intrinsics[c].focal_length),
                                    camera_intrinsics[c].principal_point,
                                    const_cast<float *>(camera_intrinsics[c].distortion),
                                    rotVect,
                                    transVect);

        // Check result against an arbitrary limit to eliminate impossible values.
        // (If the chess board were truly that far away, the camera wouldn't be able to
        // see the squares.)
        if (transVect[0] > 1000*square_size
            || transVect[1] > 1000*square_size
            || transVect[2] > 1000*square_size)
        {
            // ignore impossible results
            continue;
        }

        CvMat rotMatrDescr = cvMat(3, 3, CV_32FC1, rotMatr);
        CvMat rotVectDescr = cvMat(3, 1, CV_32FC1, rotVect);

        /* Calc rotation matrix by Rodrigues Transform */
        cvRodrigues2( &rotVectDescr, &rotMatrDescr );

        //combine the two transformations into one matrix
        //order is important! rotations are not commutative
        float tmat[4][4] = { { 1.f, 0.f, 0.f, 0.f },
                             { 0.f, 1.f, 0.f, 0.f },
                             { 0.f, 0.f, 1.f, 0.f },
                             { transVect[0], transVect[1], transVect[2], 1.f } };
        
        float rmat[4][4] = { { rotMatr[0], rotMatr[1], rotMatr[2], 0.f },
                             { rotMatr[3], rotMatr[4], rotMatr[5], 0.f },
                             { rotMatr[6], rotMatr[7], rotMatr[8], 0.f },
                             { 0.f, 0.f, 0.f, 1.f } };


        MultMatrix(camera_info[c].mat, tmat, rmat);

        // change the transformation of the cameras to put them in the world coordinate 
        // system we want to work with.

        // Start with an identity matrix; then fill in the values to accomplish
        // the desired transformation.
        float smat[4][4] = { { 1.f, 0.f, 0.f, 0.f },
                             { 0.f, 1.f, 0.f, 0.f },
                             { 0.f, 0.f, 1.f, 0.f },
                             { 0.f, 0.f, 0.f, 1.f } };

        // First, reflect through the origin by inverting all three axes.
        smat[0][0] = -1.f;
        smat[1][1] = -1.f;
        smat[2][2] = -1.f;
        MultMatrix(tmat, camera_info[c].mat, smat);

        // Scale x and y coordinates by the focal length (allowing for non-square pixels
        // and/or non-symmetrical lenses).
        smat[0][0] = 1.0f / camera_intrinsics[c].focal_length[0];
        smat[1][1] = 1.0f / camera_intrinsics[c].focal_length[1];
        smat[2][2] = 1.0f;
        MultMatrix(camera_info[c].mat, smat, tmat);

        camera_info[c].principal_point = camera_intrinsics[c].principal_point;
        camera_info[c].valid = true;

        cameras_done++;
    }

exit:
    cvReleaseImage(&gray_img);
    cvReleaseImage(&tmp_img);
    cvFree(&object_points);
    cvFree(&points);

    return cameras_done == num_cameras;
}

// fill in the real-world coordinates of the checkerboard points
static void FillObjectPoints(CvPoint3D32f *obj_points, CvSize etalon_size, float square_size)
{
    int x, y, i;

    for (y = 0, i = 0; y < etalon_size.height; y++)
    {
        for (x = 0; x < etalon_size.width; x++, i++)
        {
            obj_points[i].x = square_size * x;
            obj_points[i].y = square_size * y;
            obj_points[i].z = 0;
        }
    }
}


// Mark the points found on the input image
// The marks are drawn multi-colored if all the points were found.
static void DrawEtalon(IplImage *img, CvPoint2D32f *corners,
                       int corner_count, CvSize etalon_size, int draw_ordered)
{
    const int r = 4;
    int i;
    int x, y;
    CvPoint prev_pt = { 0, 0 };
    static const CvScalar rgb_colors[] = {
        {{0,0,255}},
        {{0,128,255}},
        {{0,200,200}},
        {{0,255,0}},
        {{200,200,0}},
        {{255,0,0}},
        {{255,0,255}} };
    static const CvScalar gray_colors[] = {
        {{80}}, {{120}}, {{160}}, {{200}}, {{100}}, {{140}}, {{180}}
    };
    const CvScalar* colors = img->nChannels == 3 ? rgb_colors : gray_colors;

    CvScalar color = colors[0];
    for (y = 0, i = 0; y < etalon_size.height; y++)
    {
        if (draw_ordered)
            color = colors[y % ARRAY_SIZEOF(rgb_colors)];

        for (x = 0; x < etalon_size.width && i < corner_count; x++, i++)
        {
            CvPoint pt;
            pt.x = cvRound(corners[i].x);
            pt.y = cvRound(corners[i].y);
            if (img->origin == IPL_ORIGIN_BL)
                pt.y = img->height - 1 - pt.y;

            if (draw_ordered)
            {
                if (i != 0)
                   cvLine(img, prev_pt, pt, color, 1, CV_AA);
                prev_pt = pt;
            }

            cvLine( img, cvPoint(pt.x - r, pt.y - r),
                    cvPoint(pt.x + r, pt.y + r), color, 1, CV_AA );
            cvLine( img, cvPoint(pt.x - r, pt.y + r),
                    cvPoint(pt.x + r, pt.y - r), color, 1, CV_AA );
            cvCircle( img, pt, r+1, color, 1, CV_AA );
        }
    }
}

// Find the midpoint of the line segment between two points.
static CvPoint3D32f midpoint(const CvPoint3D32f &p1, const CvPoint3D32f &p2)
{
    return cvPoint3D32f((p1.x+p2.x)/2, (p1.y+p2.y)/2, (p1.z+p2.z)/2);
}

static void operator +=(CvPoint3D32f &p1, const CvPoint3D32f &p2)
{
    p1.x += p2.x;
    p1.y += p2.y;
    p1.z += p2.z;
}

static CvPoint3D32f operator /(const CvPoint3D32f &p, int d)
{
    return cvPoint3D32f(p.x/d, p.y/d, p.z/d);
}

static const Cv3dTracker2dTrackedObject *find(const Cv3dTracker2dTrackedObject v[], int num_objects, int id)
{
    for (int i = 0; i < num_objects; i++)
    {
        if (v[i].id == id)
            return &v[i];
    }
    return NULL;
}

#define CAMERA_POS(c) (cvPoint3D32f((c).mat[3][0], (c).mat[3][1], (c).mat[3][2]))

//////////////////////////////
// cv3dTrackerLocateObjects //
//////////////////////////////
CV_IMPL int  cv3dTrackerLocateObjects(int num_cameras, int num_objects,
                 const Cv3dTrackerCameraInfo camera_info[],      // size is num_cameras
                 const Cv3dTracker2dTrackedObject tracking_info[], // size is num_objects*num_cameras
                 Cv3dTrackerTrackedObject tracked_objects[])     // size is num_objects
{
    /*CV_FUNCNAME("cv3dTrackerLocateObjects");*/
    int found_objects = 0;

    // count how many cameras could see each object
    std::map<int, int> count;
    for (int c = 0; c < num_cameras; c++)
    {
        if (!camera_info[c].valid)
            continue;

        for (int i = 0; i < num_objects; i++)
        {
            const Cv3dTracker2dTrackedObject *o = &tracking_info[c*num_objects+i];
            if (o->id != -1)
                count[o->id]++;
        }
    }

    // process each object that was seen by at least two cameras
    for (std::map<int, int>::iterator i = count.begin(); i != count.end(); i++)
    {
        if (i->second < 2)
            continue; // ignore object seen by only one camera
        int id = i->first;

        // find an approximation of the objects location for each pair of cameras that
        // could see this object, and average them
        CvPoint3D32f total = cvPoint3D32f(0, 0, 0);
        int weight = 0;

        for (int c1 = 0; c1 < num_cameras-1; c1++)
        {
            if (!camera_info[c1].valid)
                continue;

            const Cv3dTracker2dTrackedObject *o1 = find(&tracking_info[c1*num_objects],
                                                        num_objects, id);
            if (o1 == NULL)
                continue; // this camera didn't see this object

            CvPoint3D32f p1a = CAMERA_POS(camera_info[c1]);
            CvPoint3D32f p1b = ImageCStoWorldCS(camera_info[c1], o1->p);

            for (int c2 = c1 + 1; c2 < num_cameras; c2++)
            {
                if (!camera_info[c2].valid)
                    continue;

                const Cv3dTracker2dTrackedObject *o2 = find(&tracking_info[c2*num_objects],
                                                            num_objects, id);
                if (o2 == NULL)
                    continue; // this camera didn't see this object

                CvPoint3D32f p2a = CAMERA_POS(camera_info[c2]);
                CvPoint3D32f p2b = ImageCStoWorldCS(camera_info[c2], o2->p);

                // these variables are initialized simply to avoid erroneous error messages
                // from the compiler
                CvPoint3D32f r1 = cvPoint3D32f(0, 0, 0);
                CvPoint3D32f r2 = cvPoint3D32f(0, 0, 0);

                // find the intersection of the two lines (or the points of closest
                // approach, if they don't intersect)
                if (!intersection(p1a, p1b, p2a, p2b, r1, r2))
                    continue;

                total += midpoint(r1, r2);
                weight++;
            }
        }

        CvPoint3D32f center = total/weight;
        tracked_objects[found_objects++] = cv3dTrackerTrackedObject(id, center);
    }

    return found_objects;
}

#define EPS 1e-9

// Compute the determinant of the 3x3 matrix represented by 3 row vectors.
static inline double det(CvPoint3D32f v1, CvPoint3D32f v2, CvPoint3D32f v3)
{
    return v1.x*v2.y*v3.z + v1.z*v2.x*v3.y + v1.y*v2.z*v3.x
           - v1.z*v2.y*v3.x - v1.x*v2.z*v3.y - v1.y*v2.x*v3.z;
}

static CvPoint3D32f operator +(CvPoint3D32f a, CvPoint3D32f b)
{
    return cvPoint3D32f(a.x + b.x, a.y + b.y, a.z + b.z);
}

static CvPoint3D32f operator -(CvPoint3D32f a, CvPoint3D32f b)
{
    return cvPoint3D32f(a.x - b.x, a.y - b.y, a.z - b.z);
}

static CvPoint3D32f operator *(CvPoint3D32f v, double f)
{
    return cvPoint3D32f(f*v.x, f*v.y, f*v.z);
}


// Find the intersection of two lines, or if they don't intersect,
// the points of closest approach.
// The lines are defined by (o1,p1) and (o2, p2).
// If they intersect, r1 and r2 will be the same.
// Returns false on error.
static bool intersection(CvPoint3D32f o1, CvPoint3D32f p1,
                         CvPoint3D32f o2, CvPoint3D32f p2,
                         CvPoint3D32f &r1, CvPoint3D32f &r2)
{
    CvPoint3D32f x = o2 - o1;
    CvPoint3D32f d1 = p1 - o1;
    CvPoint3D32f d2 = p2 - o2;

    CvPoint3D32f cross = cvPoint3D32f(d1.y*d2.z - d1.z*d2.y,
                                      d1.z*d2.x - d1.x*d2.z,
                                      d1.x*d2.y - d1.y*d2.x);
    double den = cross.x*cross.x + cross.y*cross.y + cross.z*cross.z;

    if (den < EPS)
        return false;

    double t1 = det(x, d2, cross) / den;
    double t2 = det(x, d1, cross) / den;

    r1 = o1 + d1 * t1;
    r2 = o2 + d2 * t2;

    return true;
}

// Convert from image to camera space by transforming point p in
// the image plane by the camera matrix.
static CvPoint3D32f ImageCStoWorldCS(const Cv3dTrackerCameraInfo &camera_info, CvPoint2D32f p)
{
    float tp[4];
    tp[0] = (float)p.x - camera_info.principal_point.x;
    tp[1] = (float)p.y - camera_info.principal_point.y;
    tp[2] = 1.f;
    tp[3] = 1.f;

    float tr[4];
    //multiply tp by mat to get tr
    MultVectorMatrix(tr, tp, camera_info.mat);

    return cvPoint3D32f(tr[0]/tr[3], tr[1]/tr[3], tr[2]/tr[3]);
}

// Multiply affine transformation m1 by the affine transformation m2 and
// return the result in rm.
static void MultMatrix(float rm[4][4], const float m1[4][4], const float m2[4][4])
{
    for (int i=0; i<=3; i++)
        for (int j=0; j<=3; j++)
        {
            rm[i][j]= 0.0;
            for (int k=0; k <= 3; k++)
                rm[i][j] += m1[i][k]*m2[k][j];
        }
}

// Multiply the vector v by the affine transformation matrix m and return the
// result in rv.
void MultVectorMatrix(float rv[4], const float v[4], const float m[4][4])
{
    for (int i=0; i<=3; i++)
    {
        rv[i] = 0.f;
        for (int j=0;j<=3;j++)
            rv[i] += v[j] * m[j][i];
    }
}
