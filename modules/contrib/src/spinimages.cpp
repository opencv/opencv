/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
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
#include <algorithm>
#include <cmath>
#include <functional>
#include <fstream>
#include <limits>
#include <set>

using namespace cv;
using namespace std;

/********************************* local utility *********************************/

namespace cv
{
    using std::log;
    using std::max;
    using std::min;
    using std::sqrt;
}
namespace 
{
    const static Scalar colors[] = 
    {
        CV_RGB(255,   0,   0),
        CV_RGB(  0, 255,   0),
        CV_RGB(  0,   0, 255),
        CV_RGB(255, 255,   0),
        CV_RGB(255,   0, 255),
        CV_RGB(  0, 255, 255),
        CV_RGB(255, 127, 127),
        CV_RGB(127, 127, 255),
        CV_RGB(127, 255, 127),
        CV_RGB(255, 255, 127),
        CV_RGB(127, 255, 255),
        CV_RGB(255, 127, 255),
        CV_RGB(127,   0,   0),
        CV_RGB(  0, 127,   0),
        CV_RGB(  0,   0, 127),
        CV_RGB(127, 127,   0),
        CV_RGB(127,   0, 127),
        CV_RGB(  0, 127, 127)
    };
    size_t colors_mum = sizeof(colors)/sizeof(colors[0]);

template<class FwIt, class T> void iota(FwIt first, FwIt last, T value) { while(first != last) *first++ = value++; }

void computeNormals( const Octree& Octree, const vector<Point3f>& centers, vector<Point3f>& normals, 
                    vector<uchar>& mask, float normalRadius, int minNeighbors = 20)
{    
    size_t normals_size = centers.size();
    normals.resize(normals_size);
    
    if (mask.size() != normals_size)
    {
        size_t m = mask.size();        
        mask.resize(normals_size);
        if (normals_size > m)
            for(; m < normals_size; ++m)
                mask[m] = 1;
    }
    
    vector<Point3f> buffer;
    buffer.reserve(128);
    SVD svd;

    const static Point3f zero(0.f, 0.f, 0.f);

    for(size_t n = 0; n < normals_size; ++n)
    {
        if (mask[n] == 0)
            continue;

        const Point3f& center = centers[n];
        Octree.getPointsWithinSphere(center, normalRadius, buffer);

        int buf_size = (int)buffer.size();
        if (buf_size < minNeighbors)
        {
            normals[n] = Mesh3D::allzero;
            mask[n] = 0;
            continue;
        }

        //find the mean point for normalization
        Point3f mean(Mesh3D::allzero);
        for(int i = 0; i < buf_size; ++i)
            mean += buffer[i];

        mean.x /= buf_size;
        mean.y /= buf_size;
        mean.z /= buf_size;
            
        double pxpx = 0;
        double pypy = 0;
        double pzpz = 0;

        double pxpy = 0;
        double pxpz = 0;
        double pypz = 0;

        for(int i = 0; i < buf_size; ++i)
        {
            const Point3f& p = buffer[i];

            pxpx += (p.x - mean.x) * (p.x - mean.x);
            pypy += (p.y - mean.y) * (p.y - mean.y);
            pzpz += (p.z - mean.z) * (p.z - mean.z);

            pxpy += (p.x - mean.x) * (p.y - mean.y);
            pxpz += (p.x - mean.x) * (p.z - mean.z);
            pypz += (p.y - mean.y) * (p.z - mean.z);
        }

        //create and populate matrix with normalized nbrs
        double M_data[] = { pxpx, pxpy, pxpz, /**/ pxpy, pypy, pypz, /**/ pxpz, pypz, pzpz };
        Mat M(3, 3, CV_64F, M_data);

        svd(M, SVD::MODIFY_A);

        /*normals[n] = Point3f(  (float)((double*)svd.vt.data)[6],
                                 (float)((double*)svd.vt.data)[7],
                                 (float)((double*)svd.vt.data)[8]  );*/            
        normals[n] = reinterpret_cast<Point3d*>(svd.vt.data)[2];                
        mask[n] = 1;        
    }
}

void initRotationMat(const Point3f& n, float out[9])
{
    double pitch = atan2(n.x, n.z);
    double pmat[] = { cos(pitch), 0, -sin(pitch) ,
                        0      , 1,      0      ,
                     sin(pitch), 0,  cos(pitch) };

    double roll = atan2((double)n.y, n.x * pmat[3*2+0] + n.z * pmat[3*2+2]);

    double rmat[] = { 1,     0,         0,
                     0, cos(roll), -sin(roll) ,
                     0, sin(roll),  cos(roll) };

    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            out[3*i+j] = (float)(rmat[3*i+0]*pmat[3*0+j] +
                rmat[3*i+1]*pmat[3*1+j] + rmat[3*i+2]*pmat[3*2+j]);
}

void transform(const Point3f& in, float matrix[9], Point3f& out)
{
    out.x = in.x * matrix[3*0+0] + in.y * matrix[3*0+1] + in.z * matrix[3*0+2];
    out.y = in.x * matrix[3*1+0] + in.y * matrix[3*1+1] + in.z * matrix[3*1+2];
    out.z = in.x * matrix[3*2+0] + in.y * matrix[3*2+1] + in.z * matrix[3*2+2];
}

#if CV_SSE2
void convertTransformMatrix(const float* matrix, float* sseMatrix)
{
    sseMatrix[0] = matrix[0]; sseMatrix[1] = matrix[3]; sseMatrix[2] = matrix[6]; sseMatrix[3] = 0;
    sseMatrix[4] = matrix[1]; sseMatrix[5] = matrix[4]; sseMatrix[6] = matrix[7]; sseMatrix[7] = 0;
    sseMatrix[8] = matrix[2]; sseMatrix[9] = matrix[5]; sseMatrix[10] = matrix[8]; sseMatrix[11] = 0;
}

inline __m128 transformSSE(const __m128* matrix, const __m128& in)
{
    assert(((size_t)matrix & 15) == 0);
    __m128 a0 = _mm_mul_ps(_mm_load_ps((float*)(matrix+0)), _mm_shuffle_ps(in,in,_MM_SHUFFLE(0,0,0,0)));
    __m128 a1 = _mm_mul_ps(_mm_load_ps((float*)(matrix+1)), _mm_shuffle_ps(in,in,_MM_SHUFFLE(1,1,1,1)));
    __m128 a2 = _mm_mul_ps(_mm_load_ps((float*)(matrix+2)), _mm_shuffle_ps(in,in,_MM_SHUFFLE(2,2,2,2)));

    return _mm_add_ps(_mm_add_ps(a0,a1),a2);
}

inline __m128i _mm_mullo_epi32_emul(const __m128i& a, __m128i& b)
{    
    __m128i pack = _mm_packs_epi32(a, a);
    return _mm_unpacklo_epi16(_mm_mullo_epi16(pack, b), _mm_mulhi_epi16(pack, b));    
}

#endif

void computeSpinImages( const Octree& Octree, const vector<Point3f>& points, const vector<Point3f>& normals, 
                       vector<uchar>& mask, Mat& spinImages, int imageWidth, float binSize)
{   
    float pixelsPerMeter = 1.f / binSize;
    float support = imageWidth * binSize;    
    
    assert(normals.size() == points.size());
    assert(mask.size() == points.size());
    
    size_t points_size = points.size();
    mask.resize(points_size);

    int height = imageWidth;
    int width  = imageWidth;

    spinImages.create( (int)points_size, width*height, CV_32F );

    int nthreads = getNumThreads();
    int i;

    vector< vector<Point3f> > pointsInSpherePool(nthreads);
    for(i = 0; i < nthreads; i++)
        pointsInSpherePool[i].reserve(2048);

    float halfSuppport = support / 2;
    float searchRad = support * sqrt(5.f) / 2;  //  sqrt(sup*sup + (sup/2) * (sup/2) )

#ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads)
#endif
    for(i = 0; i < (int)points_size; ++i)
    {
        if (mask[i] == 0)
            continue;

        int t = cvGetThreadNum();
        vector<Point3f>& pointsInSphere = pointsInSpherePool[t];
                
        const Point3f& center = points[i];
        Octree.getPointsWithinSphere(center, searchRad, pointsInSphere);

        size_t inSphere_size = pointsInSphere.size();
        if (inSphere_size == 0)
        {
            mask[i] = 0;
            continue;
        }

        const Point3f& normal = normals[i];
        
        float rotmat[9];
        initRotationMat(normal, rotmat);
        Point3f new_center;
        transform(center, rotmat, new_center);

        Mat spinImage = spinImages.row(i).reshape(1, height);
        float* spinImageData = (float*)spinImage.data;
        int step = width;
        spinImage = Scalar(0.);

        float alpha, beta;
        size_t j = 0;
#if CV_SSE2
        if (inSphere_size > 4 && checkHardwareSupport(CV_CPU_SSE2))
        {
            __m128 rotmatSSE[3];
            convertTransformMatrix(rotmat, (float*)rotmatSSE);
            
            __m128 center_x4 = _mm_set1_ps(new_center.x);
            __m128 center_y4 = _mm_set1_ps(new_center.y);
            __m128 center_z4 = _mm_set1_ps(new_center.z + halfSuppport);
            __m128 ppm4 = _mm_set1_ps(pixelsPerMeter);
            __m128i height4m1 = _mm_set1_epi32(spinImage.rows-1);
            __m128i width4m1 = _mm_set1_epi32(spinImage.cols-1);
            assert( spinImage.step <= 0xffff );
            __m128i step4 = _mm_set1_epi16((short)step);
            __m128i zero4 = _mm_setzero_si128();
            __m128i one4i = _mm_set1_epi32(1);
            __m128 zero4f = _mm_setzero_ps();
            __m128 one4f = _mm_set1_ps(1.f);
            //__m128 two4f = _mm_set1_ps(2.f);
            int CV_DECL_ALIGNED(16) o[4];

            for (; j <= inSphere_size - 5; j += 4)
            {
                __m128 pt0 = transformSSE(rotmatSSE, _mm_loadu_ps((float*)&pointsInSphere[j+0])); // x0 y0 z0 .
                __m128 pt1 = transformSSE(rotmatSSE, _mm_loadu_ps((float*)&pointsInSphere[j+1])); // x1 y1 z1 .
                __m128 pt2 = transformSSE(rotmatSSE, _mm_loadu_ps((float*)&pointsInSphere[j+2])); // x2 y2 z2 .
                __m128 pt3 = transformSSE(rotmatSSE, _mm_loadu_ps((float*)&pointsInSphere[j+3])); // x3 y3 z3 .

                __m128 z0 = _mm_unpackhi_ps(pt0, pt1); // z0 z1 . .
                __m128 z1 = _mm_unpackhi_ps(pt2, pt3); // z2 z3 . .
                __m128 beta4 = _mm_sub_ps(center_z4, _mm_movelh_ps(z0, z1)); // b0 b1 b2 b3
                
                __m128 xy0 = _mm_unpacklo_ps(pt0, pt1); // x0 x1 y0 y1
                __m128 xy1 = _mm_unpacklo_ps(pt2, pt3); // x2 x3 y2 y3
                __m128 x4 = _mm_movelh_ps(xy0, xy1); // x0 x1 x2 x3
                __m128 y4 = _mm_movehl_ps(xy1, xy0); // y0 y1 y2 y3

                x4 = _mm_sub_ps(x4, center_x4);
                y4 = _mm_sub_ps(y4, center_y4);
                __m128 alpha4 = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(x4,x4),_mm_mul_ps(y4,y4)));
                
                __m128 n1f4 = _mm_mul_ps( beta4, ppm4);  /* beta4 float */
                __m128 n2f4 = _mm_mul_ps(alpha4, ppm4); /* alpha4 float */

                /* floor */
                __m128i n1 = _mm_sub_epi32(_mm_cvttps_epi32( _mm_add_ps( n1f4, one4f ) ), one4i);
                __m128i n2 = _mm_sub_epi32(_mm_cvttps_epi32( _mm_add_ps( n2f4, one4f ) ), one4i);

                __m128 f1 = _mm_sub_ps( n1f4, _mm_cvtepi32_ps(n1) );  /* { beta4  }  */
                __m128 f2 = _mm_sub_ps( n2f4, _mm_cvtepi32_ps(n2) );  /* { alpha4 }  */

                __m128 f1f2 = _mm_mul_ps(f1, f2);  // f1 * f2                        
                __m128 omf1omf2 = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(one4f, f2), f1), f1f2); // (1-f1) * (1-f2)
                
                __m128i mask = _mm_and_si128(
                    _mm_andnot_si128(_mm_cmpgt_epi32(zero4, n1), _mm_cmpgt_epi32(height4m1, n1)),
                    _mm_andnot_si128(_mm_cmpgt_epi32(zero4, n2), _mm_cmpgt_epi32(width4m1, n2)));

                __m128 maskf = _mm_cmpneq_ps(_mm_cvtepi32_ps(mask), zero4f);
                            
                __m128 v00 = _mm_and_ps(        omf1omf2       , maskf); // a00 b00 c00 d00
                __m128 v01 = _mm_and_ps( _mm_sub_ps( f2, f1f2 ), maskf); // a01 b01 c01 d01
                __m128 v10 = _mm_and_ps( _mm_sub_ps( f1, f1f2 ), maskf); // a10 b10 c10 d10
                __m128 v11 = _mm_and_ps(          f1f2         , maskf); // a11 b11 c11 d11

                __m128i ofs4 = _mm_and_si128(_mm_add_epi32(_mm_mullo_epi32_emul(n1, step4), n2), mask);
                _mm_store_si128((__m128i*)o, ofs4);

                __m128 t0 = _mm_unpacklo_ps(v00, v01); // a00 a01 b00 b01
                __m128 t1 = _mm_unpacklo_ps(v10, v11); // a10 a11 b10 b11
                __m128 u0 = _mm_movelh_ps(t0, t1); // a00 a01 a10 a11
                __m128 u1 = _mm_movehl_ps(t1, t0); // b00 b01 b10 b11

                __m128 x0 = _mm_loadl_pi(u0, (__m64*)(spinImageData+o[0])); // x00 x01
                x0 = _mm_loadh_pi(x0, (__m64*)(spinImageData+o[0]+step));   // x00 x01 x10 x11
                x0 = _mm_add_ps(x0, u0);
                _mm_storel_pi((__m64*)(spinImageData+o[0]), x0);
                _mm_storeh_pi((__m64*)(spinImageData+o[0]+step), x0);

                x0 = _mm_loadl_pi(x0, (__m64*)(spinImageData+o[1]));        // y00 y01
                x0 = _mm_loadh_pi(x0, (__m64*)(spinImageData+o[1]+step));   // y00 y01 y10 y11
                x0 = _mm_add_ps(x0, u1);
                _mm_storel_pi((__m64*)(spinImageData+o[1]), x0);
                _mm_storeh_pi((__m64*)(spinImageData+o[1]+step), x0);

                t0 = _mm_unpackhi_ps(v00, v01); // c00 c01 d00 d01
                t1 = _mm_unpackhi_ps(v10, v11); // c10 c11 d10 d11
                u0 = _mm_movelh_ps(t0, t1); // c00 c01 c10 c11
                u1 = _mm_movehl_ps(t1, t0); // d00 d01 d10 d11

                x0 = _mm_loadl_pi(x0, (__m64*)(spinImageData+o[2]));        // z00 z01
                x0 = _mm_loadh_pi(x0, (__m64*)(spinImageData+o[2]+step));   // z00 z01 z10 z11
                x0 = _mm_add_ps(x0, u0);
                _mm_storel_pi((__m64*)(spinImageData+o[2]), x0);
                _mm_storeh_pi((__m64*)(spinImageData+o[2]+step), x0);

                x0 = _mm_loadl_pi(x0, (__m64*)(spinImageData+o[3]));        // w00 w01
                x0 = _mm_loadh_pi(x0, (__m64*)(spinImageData+o[3]+step));   // w00 w01 w10 w11
                x0 = _mm_add_ps(x0, u1);
                _mm_storel_pi((__m64*)(spinImageData+o[3]), x0);
                _mm_storeh_pi((__m64*)(spinImageData+o[3]+step), x0);
            }
        }
#endif
        for (; j < inSphere_size; ++j)
        {
            Point3f pt;
            transform(pointsInSphere[j], rotmat, pt);

            beta = halfSuppport - (pt.z - new_center.z);
            if (beta >= support || beta < 0)
                continue;

            alpha = sqrt( (new_center.x - pt.x) * (new_center.x - pt.x) + 
                          (new_center.y - pt.y) * (new_center.y - pt.y) ); 
            
            float n1f = beta  * pixelsPerMeter;
            float n2f = alpha * pixelsPerMeter;

            int n1 = cvFloor(n1f);
            int n2 = cvFloor(n2f);

            float f1 = n1f - n1;
            float f2 = n2f - n2;

            if  ((unsigned)n1 >= (unsigned)(spinImage.rows-1) || 
                 (unsigned)n2 >= (unsigned)(spinImage.cols-1))
                continue;

            float *cellptr = spinImageData + step * n1 + n2;
            float f1f2 = f1*f2;
            cellptr[0] += 1 - f1 - f2 + f1f2;
            cellptr[1] += f2 - f1f2;
            cellptr[step] += f1 - f1f2;
            cellptr[step+1] += f1f2;
        }
        mask[i] = 1;
    }
}

}

/********************************* Mesh3D *********************************/

const Point3f cv::Mesh3D::allzero(0.f, 0.f, 0.f);

cv::Mesh3D::Mesh3D() { resolution = -1; }
cv::Mesh3D::Mesh3D(const vector<Point3f>& _vtx)
{
    resolution = -1;
    vtx.resize(_vtx.size());
    std::copy(_vtx.begin(), _vtx.end(), vtx.begin());
}
cv::Mesh3D::~Mesh3D() {}

void cv::Mesh3D::buildOctree() { if (octree.getNodes().empty()) octree.buildTree(vtx); }
void cv::Mesh3D::clearOctree(){ octree = Octree(); }

float cv::Mesh3D::estimateResolution(float tryRatio)
{
    const int neighbors = 3;
    const int minReasonable = 10;

    int tryNum = static_cast<int>(tryRatio * vtx.size());
    tryNum = min(max(tryNum, minReasonable), (int)vtx.size());

    CvMat desc = cvMat((int)vtx.size(), 3, CV_32F, &vtx[0]);
    CvFeatureTree* tr = cvCreateKDTree(&desc);

    vector<double> dist(tryNum * neighbors);
    vector<int>    inds(tryNum * neighbors);
    vector<Point3f> query;  

    RNG& rng = theRNG();          
    for(int i = 0; i < tryNum; ++i)
        query.push_back(vtx[rng.next() % vtx.size()]);
        
    CvMat cvinds  = cvMat( (int)tryNum, neighbors, CV_32S,  &inds[0] );
    CvMat cvdist  = cvMat( (int)tryNum, neighbors, CV_64F,  &dist[0] );    
    CvMat cvquery = cvMat( (int)tryNum,         3, CV_32F, &query[0] );
    cvFindFeatures(tr, &cvquery, &cvinds, &cvdist, neighbors, 50);    
    cvReleaseFeatureTree(tr);

    const int invalid_dist = -2;    
    for(int i = 0; i < tryNum; ++i)
        if (inds[i] == -1)
            dist[i] = invalid_dist;

    dist.resize(remove(dist.begin(), dist.end(), invalid_dist) - dist.begin());
        
    sort(dist, less<double>());
   
    return resolution = (float)dist[ dist.size() / 2 ];
}

void cv::Mesh3D::computeNormals(float normalRadius, int minNeighbors)
{
    buildOctree();
    vector<uchar> mask;
    ::computeNormals(octree, vtx, normals, mask, normalRadius, minNeighbors);
}

void cv::Mesh3D::computeNormals(const vector<int>& subset, float normalRadius, int minNeighbors)
{
    buildOctree();
    vector<uchar> mask(vtx.size(), 0);
    for(size_t i = 0; i < subset.size(); ++i) 
        mask[subset[i]] = 1;
    ::computeNormals(octree, vtx, normals, mask, normalRadius, minNeighbors);
}

void cv::Mesh3D::writeAsVrml(const String& file, const vector<Scalar>& colors) const
{
    ofstream ofs(file.c_str());

    ofs << "#VRML V2.0 utf8" << endl;
	ofs << "Shape" << std::endl << "{" << endl;
	ofs << "geometry PointSet" << endl << "{" << endl;
	ofs << "coord Coordinate" << endl << "{" << endl;
	ofs << "point[" << endl;

    for(size_t i = 0; i < vtx.size(); ++i)
        ofs << vtx[i].x << " " << vtx[i].y << " " << vtx[i].z << endl;
    
	ofs << "]" << endl; //point[
	ofs << "}" << endl; //Coordinate{

    if (vtx.size() == colors.size())
    {
        ofs << "color Color" << endl << "{" << endl;
        ofs << "color[" << endl;
    	
        for(size_t i = 0; i < colors.size(); ++i)
            ofs << (float)colors[i][2] << " " << (float)colors[i][1] << " " << (float)colors[i][0] << endl;        
      
        ofs << "]" << endl; //color[
	    ofs << "}" << endl; //color Color{
    }

	ofs << "}" << endl; //PointSet{
	ofs << "}" << endl; //Shape{
}


/********************************* SpinImageModel *********************************/


bool cv::SpinImageModel::spinCorrelation(const Mat& spin1, const Mat& spin2, float lambda, float& result)
{
    struct Math { static double atanh(double x) { return 0.5 * std::log( (1 + x) / (1 - x) ); } };
      
    const float* s1 = spin1.ptr<float>();
    const float* s2 = spin2.ptr<float>();

    int spin_sz = spin1.cols * spin1.rows; 
    double sum1 = 0.0, sum2 = 0.0, sum12 = 0.0, sum11 = 0.0, sum22 = 0.0;

    int N = 0;
    int i = 0;
#if CV_SSE2//____________TEMPORARY_DISABLED_____________
    float CV_DECL_ALIGNED(16) su1[4], su2[4], su11[4], su22[4], su12[4], n[4];    
    
    __m128 zerof4 = _mm_setzero_ps();
    __m128 onef4  = _mm_set1_ps(1.f);
    __m128 Nf4 = zerof4;    
    __m128 sum1f4  = zerof4;
    __m128 sum2f4  = zerof4;
    __m128 sum11f4 = zerof4;
    __m128 sum22f4 = zerof4;
    __m128 sum12f4 = zerof4;        
    for(; i < spin_sz - 5; i += 4)
    {
        __m128 v1f4 = _mm_loadu_ps(s1 + i); 
        __m128 v2f4 = _mm_loadu_ps(s2 + i); 

        __m128 mskf4 = _mm_and_ps(_mm_cmpneq_ps(v1f4, zerof4), _mm_cmpneq_ps(v2f4, zerof4));
        if( !_mm_movemask_ps(mskf4) ) 
            continue;
        
        Nf4 = _mm_add_ps(Nf4, _mm_and_ps(onef4, mskf4));

        v1f4 = _mm_and_ps(v1f4, mskf4);
        v2f4 = _mm_and_ps(v2f4, mskf4);
     
        sum1f4 = _mm_add_ps(sum1f4, v1f4);
        sum2f4 = _mm_add_ps(sum2f4, v2f4);
        sum11f4 = _mm_add_ps(sum11f4, _mm_mul_ps(v1f4, v1f4));
        sum22f4 = _mm_add_ps(sum22f4, _mm_mul_ps(v2f4, v2f4));
        sum12f4 = _mm_add_ps(sum12f4, _mm_mul_ps(v1f4, v2f4));        
    }
    _mm_store_ps( su1,  sum1f4 );
    _mm_store_ps( su2,  sum2f4 );
    _mm_store_ps(su11, sum11f4 );
    _mm_store_ps(su22, sum22f4 );
    _mm_store_ps(su12, sum12f4 );
    _mm_store_ps(n, Nf4 );

    N = static_cast<int>(n[0] + n[1] + n[2] + n[3]);
    sum1  =  su1[0] +  su1[1] +  su1[2] +  su1[3];
    sum2  =  su2[0] +  su2[1] +  su2[2] +  su2[3];
    sum11 = su11[0] + su11[1] + su11[2] + su11[3];
    sum22 = su22[0] + su22[1] + su22[2] + su22[3];
    sum12 = su12[0] + su12[1] + su12[2] + su12[3];
#endif

    for(; i < spin_sz; ++i)
    {
        float v1 = s1[i];
        float v2 = s2[i];

        if( !v1 || !v2 )
            continue;
        N++;
     
        sum1  += v1; 
        sum2  += v2; 
        sum11 += v1 * v1; 
        sum22 += v2 * v2; 
        sum12 += v1 * v2;
    }
    if( N < 4 )
        return false;

    double sum1sum1 = sum1 * sum1;
    double sum2sum2 = sum2 * sum2;

    double Nsum12 = N * sum12;
    double Nsum11 = N * sum11;
    double Nsum22 = N * sum22;

    if (Nsum11 == sum1sum1 || Nsum22 == sum2sum2)
        return false;

    double corr = (Nsum12 - sum1 * sum2) / sqrt( (Nsum11 - sum1sum1) * (Nsum22 - sum2sum2) );
    double atanh = Math::atanh(corr);
    result = (float)( atanh * atanh - lambda * ( 1.0 / (N - 3) ) );
    return true;        
}

inline Point2f cv::SpinImageModel::calcSpinMapCoo(const Point3f& p, const Point3f& v, const Point3f& n)
{   
    /*Point3f PmV(p.x - v.x, p.y - v.y, p.z - v.z);    
    float normalNorm = (float)norm(n);    
    float beta = PmV.dot(n) / normalNorm;
    float pmcNorm = (float)norm(PmV);
    float alpha = sqrt( pmcNorm * pmcNorm - beta * beta);
    return Point2f(alpha, beta);*/

    float pmv_x = p.x - v.x, pmv_y = p.y - v.y, pmv_z = p.z - v.z;

    float beta = (pmv_x * n.x + pmv_y + n.y + pmv_z * n.z) / sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    float alpha = sqrt( pmv_x * pmv_x + pmv_y * pmv_y + pmv_z * pmv_z - beta * beta);        
    return Point2f(alpha, beta);
}

inline float cv::SpinImageModel::geometricConsistency(const Point3f& pointScene1, const Point3f& normalScene1,
                                                      const Point3f& pointModel1, const Point3f& normalModel1,
                                                      const Point3f& pointScene2, const Point3f& normalScene2,                               
                                                      const Point3f& pointModel2, const Point3f& normalModel2)
{   
    Point2f Sm2_to_m1, Ss2_to_s1;
    Point2f Sm1_to_m2, Ss1_to_s2;

    double n_Sm2_to_m1 = norm(Sm2_to_m1 = calcSpinMapCoo(pointModel2, pointModel1, normalModel1));
    double n_Ss2_to_s1 = norm(Ss2_to_s1 = calcSpinMapCoo(pointScene2, pointScene1, normalScene1));   

    double gc21 = 2 * norm(Sm2_to_m1 - Ss2_to_s1) / (n_Sm2_to_m1 + n_Ss2_to_s1 ) ;
        
    double n_Sm1_to_m2 = norm(Sm1_to_m2 = calcSpinMapCoo(pointModel1, pointModel2, normalModel2));
    double n_Ss1_to_s2 = norm(Ss1_to_s2 = calcSpinMapCoo(pointScene1, pointScene2, normalScene2));

    double gc12 = 2 * norm(Sm1_to_m2 - Ss1_to_s2) / (n_Sm1_to_m2 + n_Ss1_to_s2 ) ;

    return (float)max(gc12, gc21);
}

inline float cv::SpinImageModel::groupingCreteria(const Point3f& pointScene1, const Point3f& normalScene1,
                                                  const Point3f& pointModel1, const Point3f& normalModel1,
                                                  const Point3f& pointScene2, const Point3f& normalScene2,                               
                                                  const Point3f& pointModel2, const Point3f& normalModel2, 
                                                  float gamma)
{   
    Point2f Sm2_to_m1, Ss2_to_s1;
    Point2f Sm1_to_m2, Ss1_to_s2;

    float gamma05_inv =  0.5f/gamma;

    double n_Sm2_to_m1 = norm(Sm2_to_m1 = calcSpinMapCoo(pointModel2, pointModel1, normalModel1));
    double n_Ss2_to_s1 = norm(Ss2_to_s1 = calcSpinMapCoo(pointScene2, pointScene1, normalScene1));

    double gc21 = 2 * norm(Sm2_to_m1 - Ss2_to_s1) / (n_Sm2_to_m1 + n_Ss2_to_s1 );
    double wgc21 = gc21 / (1 - exp( -(n_Sm2_to_m1 + n_Ss2_to_s1) * gamma05_inv ) );
    
    double n_Sm1_to_m2 = norm(Sm1_to_m2 = calcSpinMapCoo(pointModel1, pointModel2, normalModel2));
    double n_Ss1_to_s2 = norm(Ss1_to_s2 = calcSpinMapCoo(pointScene1, pointScene2, normalScene2));

    double gc12 = 2 * norm(Sm1_to_m2 - Ss1_to_s2) / (n_Sm1_to_m2 + n_Ss1_to_s2 );
    double wgc12 = gc12 / (1 - exp( -(n_Sm1_to_m2 + n_Ss1_to_s2) * gamma05_inv ) );

    return (float)max(wgc12, wgc21);
}


cv::SpinImageModel::SpinImageModel(const Mesh3D& _mesh) : mesh(_mesh) , out(0)
{ 
     if (mesh.vtx.empty())
         throw Mesh3D::EmptyMeshException();
    defaultParams(); 
}
cv::SpinImageModel::SpinImageModel() : out(0) { defaultParams(); }
cv::SpinImageModel::~SpinImageModel() {}

void cv::SpinImageModel::setLogger(ostream* log) { out = log; }

void cv::SpinImageModel::defaultParams()
{
    normalRadius = 0.f;
    minNeighbors = 20;

    binSize    = 0.f; /* autodetect according to mesh resolution */
    imageWidth = 32;    
   
    lambda = 0.f; /* autodetect according to medan non zero images bin */
    gamma  = 0.f; /* autodetect according to mesh resolution */

    T_GeometriccConsistency = 0.25f;
    T_GroupingCorespondances = 0.25f;
};

Mat cv::SpinImageModel::packRandomScaledSpins(bool separateScale, size_t xCount, size_t yCount) const
{
    int spinNum = (int)getSpinCount();
    int num = min(spinNum, (int)(xCount * yCount));

    if (num == 0)
        return Mat();

    RNG& rng = theRNG();    

    vector<Mat> spins;
    for(int i = 0; i < num; ++i)
        spins.push_back(getSpinImage( rng.next() % spinNum ).reshape(1, imageWidth));    
    
    if (separateScale)
        for(int i = 0; i < num; ++i)
        {
            double max;
            Mat spin8u;
            minMaxLoc(spins[i], 0, &max);         
            spins[i].convertTo(spin8u, CV_8U, -255.0/max, 255.0);
            spins[i] = spin8u;
        }
    else
    {    
        double totalMax = 0;
        for(int i = 0; i < num; ++i)
        {
            double m;
            minMaxLoc(spins[i], 0, &m);  
            totalMax = max(m, totalMax);
        }

        for(int i = 0; i < num; ++i)
        {
            Mat spin8u;
            spins[i].convertTo(spin8u, CV_8U, -255.0/totalMax, 255.0);
            spins[i] = spin8u;
        }
    }

    int sz = spins.front().cols;

    Mat result((int)(yCount * sz + (yCount - 1)), (int)(xCount * sz + (xCount - 1)), CV_8UC3);    
    result = colors[(static_cast<int64>(cvGetTickCount()/cvGetTickFrequency())/1000) % colors_mum];

    int pos = 0;
    for(int y = 0; y < (int)yCount; ++y)
        for(int x = 0; x < (int)xCount; ++x)        
            if (pos < num)
            {
                int starty = (y + 0) * sz + y;
                int endy   = (y + 1) * sz + y;

                int startx = (x + 0) * sz + x;
                int endx   = (x + 1) * sz + x;

                Mat color;
                cvtColor(spins[pos++], color, CV_GRAY2BGR);
                Mat roi = result(Range(starty, endy), Range(startx, endx));
                color.copyTo(roi);
            } 
    return result;
}

void cv::SpinImageModel::selectRandomSubset(float ratio)
{
    ratio = min(max(ratio, 0.f), 1.f);

    size_t vtxSize = mesh.vtx.size();
    size_t setSize  = static_cast<size_t>(vtxSize * ratio);

    if (setSize == 0)
    {
        subset.clear();
    }
    else if (setSize == vtxSize)
    {
        subset.resize(vtxSize);
        iota(subset.begin(), subset.end(), 0);
    }
    else
    {
        RNG& rnd = theRNG();

        vector<size_t> left(vtxSize);
        iota(left.begin(), left.end(), (size_t)0);

        subset.resize(setSize);
        for(size_t i = 0; i < setSize; ++i)
        {
            int pos = rnd.next() % left.size();
            subset[i] = (int)left[pos];

            left[pos] = left.back();        
            left.resize(left.size() - 1);        
        }
        sort(subset, less<int>());
    }
}

void cv::SpinImageModel::setSubset(const vector<int>& ss)
{
    subset = ss;
}

void cv::SpinImageModel::repackSpinImages(const vector<uchar>& mask, Mat& spinImages, bool reAlloc) const
{    
    if (reAlloc)
    {
        size_t spinCount = mask.size() - count(mask.begin(), mask.end(), (uchar)0);
        Mat newImgs((int)spinCount, spinImages.cols, spinImages.type());    

        int pos = 0;
        for(size_t t = 0; t < mask.size(); ++t)
            if (mask[t])
            {
                Mat row = newImgs.row(pos++);
                spinImages.row((int)t).copyTo(row);
            }
        spinImages = newImgs;
    }
    else
    {
        int last = (int)mask.size();

        int dest = (int)(find(mask.begin(), mask.end(), (uchar)0) - mask.begin());
        if (dest == last)
            return;

        int first = dest + 1;
        for (; first != last; ++first)
		    if (mask[first] != 0)
            {
                Mat row = spinImages.row(dest);
                spinImages.row(first).copyTo(row);
                ++dest;
            }
        spinImages = spinImages.rowRange(0, dest);
    }
}

void cv::SpinImageModel::compute()
{
    /* estimate binSize */
    if (binSize == 0.f)
    {
         if (mesh.resolution == -1.f)
            mesh.estimateResolution();        
        binSize = mesh.resolution;
    }
    /* estimate normalRadius */    
    normalRadius = normalRadius != 0.f ? normalRadius : binSize * imageWidth / 2;    

    mesh.buildOctree();  
    if (subset.empty())
    {
        mesh.computeNormals(normalRadius, minNeighbors);
        subset.resize(mesh.vtx.size());
        iota(subset.begin(), subset.end(), 0);
    }
    else
        mesh.computeNormals(subset, normalRadius, minNeighbors);

    vector<uchar> mask(mesh.vtx.size(), 0);       
    for(size_t i = 0; i < subset.size(); ++i)
        if (mesh.normals[subset[i]] == Mesh3D::allzero)                   
            subset[i] = -1;                    
        else
            mask[subset[i]] = 1;
    subset.resize( remove(subset.begin(), subset.end(), -1) - subset.begin() );
        
    vector<Point3f> vtx;
    vector<Point3f> normals;    
    for(size_t i = 0; i < mask.size(); ++i)
        if(mask[i])
        {
            vtx.push_back(mesh.vtx[i]);
            normals.push_back(mesh.normals[i]);
        }

    vector<uchar> spinMask(vtx.size(), 1);
    computeSpinImages( mesh.octree, vtx, normals, spinMask, spinImages, imageWidth, binSize);
    repackSpinImages(spinMask, spinImages);

    size_t mask_pos = 0;
    for(size_t i = 0; i < mask.size(); ++i)
        if(mask[i])
            if (spinMask[mask_pos++] == 0)
                subset.resize( remove(subset.begin(), subset.end(), (int)i) - subset.begin() );   
}

void cv::SpinImageModel::matchSpinToModel(const Mat& spin, vector<int>& indeces, vector<float>& corrCoeffs, bool useExtremeOutliers) const
{
    const SpinImageModel& model = *this;

    indeces.clear();
    corrCoeffs.clear();

    vector<float> corrs(model.spinImages.rows);
    vector<uchar>  masks(model.spinImages.rows);
    vector<float> cleanCorrs;
    cleanCorrs.reserve(model.spinImages.rows);
    
    for(int i = 0; i < model.spinImages.rows; ++i)
    {
        masks[i] = spinCorrelation(spin, model.spinImages.row(i), model.lambda, corrs[i]);   
        if (masks[i])
            cleanCorrs.push_back(corrs[i]);
    }
    
    /* Filtering by measure histogram */
    size_t total = cleanCorrs.size();
    if(total < 5)
        return;

    sort(cleanCorrs, less<float>());
    
    float lower_fourth = cleanCorrs[(1 * total) / 4 - 1];
    float upper_fourth = cleanCorrs[(3 * total) / 4 - 0];
    float fourth_spread = upper_fourth - lower_fourth;

    //extreme or moderate?
    float coef = useExtremeOutliers ? 3.0f : 1.5f; 

    float histThresHi = upper_fourth + coef * fourth_spread;  
    //float histThresLo = lower_fourth - coef * fourth_spread; 
    
    for(size_t i = 0; i < corrs.size(); ++i)
        if (masks[i])
            if (/* corrs[i] < histThresLo || */ corrs[i] > histThresHi)
            {
                indeces.push_back((int)i);
                corrCoeffs.push_back(corrs[i]);                
            }
} 

namespace 
{

struct Match
{
    int sceneInd;        
    int modelInd;
    float measure;

    Match(){}
    Match(int sceneIndex, int modelIndex, float coeff) : sceneInd(sceneIndex), modelInd(modelIndex), measure(coeff) {}
    operator float() const { return measure; }
};

typedef set<size_t> group_t;
typedef group_t::iterator iter;
typedef group_t::const_iterator citer;

struct WgcHelper
{
    const group_t& grp;
    const Mat& mat;
    WgcHelper(const group_t& group, const Mat& groupingMat) : grp(group), mat(groupingMat){}
    float operator()(size_t leftInd) const { return Wgc(leftInd, grp); }

    /* Wgc( correspondence_C, group_{C1..Cn} ) = max_i=1..n_( Wgc(C, Ci) ) */
    float Wgc(const size_t corespInd, const group_t& group) const
    {
        const float* wgcLine = mat.ptr<float>((int)corespInd);
        float maximum = numeric_limits<float>::min();
        
        for(citer pos = group.begin(); pos != group.end(); ++pos)
            maximum = max(wgcLine[*pos], maximum);

        return maximum;
    }
private:
    WgcHelper& operator=(const WgcHelper& helper);
};

}

 void cv::SpinImageModel::match(const SpinImageModel& scene, vector< vector<Vec2i> >& result)
{   
    if (mesh.vtx.empty())
        throw Mesh3D::EmptyMeshException();

    result.clear();

    SpinImageModel& model = *this;
    const float infinity = numeric_limits<float>::infinity();
    const float float_max = numeric_limits<float>::max();
    
    /* estimate gamma */
    if (model.gamma == 0.f)
    {
        if (model.mesh.resolution == -1.f)
            model.mesh.estimateResolution();        
        model.gamma = 4 * model.mesh.resolution;
    }

    /* estimate lambda */
    if (model.lambda == 0.f)
    {
        vector<int> nonzero(model.spinImages.rows);        
        for(int i = 0; i < model.spinImages.rows; ++i)
            nonzero[i] = countNonZero(model.spinImages.row(i));
        sort(nonzero, less<int>());
        model.lambda = static_cast<float>( nonzero[ nonzero.size()/2 ] ) / 2;
    }    
       
    TickMeter corr_timer;
    corr_timer.start();
    vector<Match> allMatches;
    for(int i = 0; i < scene.spinImages.rows; ++i)
    {
        vector<int> indeces;
        vector<float> coeffs;
        matchSpinToModel(scene.spinImages.row(i), indeces, coeffs);        
        for(size_t t = 0; t < indeces.size(); ++t)
            allMatches.push_back(Match(i, indeces[t], coeffs[t])); 

        if (out) if (i % 100 == 0) *out << "Comparing scene spinimage " << i << " of " << scene.spinImages.rows << endl;        
    }
    corr_timer.stop();
    if (out) *out << "Spin correlation time  = " << corr_timer << endl;
    if (out) *out << "Matches number = " << allMatches.size() << endl;

    if(allMatches.empty())    
        return;
           
    /* filtering by similarity measure */
    const float fraction = 0.5f;
    float maxMeasure = max_element(allMatches.begin(), allMatches.end(), less<float>())->measure;    
    allMatches.erase(
        remove_if(allMatches.begin(), allMatches.end(), bind2nd(less<float>(), maxMeasure * fraction)), 
        allMatches.end());
    if (out) *out << "Matches number [filtered by similarity measure] = " << allMatches.size() << endl;

    int matchesSize = (int)allMatches.size();
    if(matchesSize == 0)
        return;
    
    /* filtering by geometric consistency */        
    for(int i = 0; i < matchesSize; ++i)
    {
        int consistNum = 1;
        float gc = float_max;
        
        for(int j = 0; j < matchesSize; ++j)
            if (i != j)
            {
                const Match& mi = allMatches[i];
                const Match& mj = allMatches[j];

                if (mi.sceneInd == mj.sceneInd || mi.modelInd == mj.modelInd)
                    gc = float_max;
                else
                {
                    const Point3f& pointSceneI  = scene.getSpinVertex(mi.sceneInd);
                    const Point3f& normalSceneI = scene.getSpinNormal(mi.sceneInd);
                
                    const Point3f& pointModelI  = model.getSpinVertex(mi.modelInd);
                    const Point3f& normalModelI = model.getSpinNormal(mi.modelInd);
                
                    const Point3f& pointSceneJ  = scene.getSpinVertex(mj.sceneInd);
                    const Point3f& normalSceneJ = scene.getSpinNormal(mj.sceneInd);
                
                    const Point3f& pointModelJ  = model.getSpinVertex(mj.modelInd);
                    const Point3f& normalModelJ = model.getSpinNormal(mj.modelInd);
             
                    gc = geometricConsistency(pointSceneI, normalSceneI, pointModelI, normalModelI,
                                              pointSceneJ, normalSceneJ, pointModelJ, normalModelJ);                                
                }

                if (gc < model.T_GeometriccConsistency)
                    ++consistNum;
            }
                    
            
        if (consistNum < matchesSize / 4) /* failed consistensy test */
            allMatches[i].measure = infinity;     
    }
    allMatches.erase(
      remove_if(allMatches.begin(), allMatches.end(), bind2nd(equal_to<float>(), infinity)), 
      allMatches.end()); 
    if (out) *out << "Matches number [filtered by geometric consistency] = " << allMatches.size() << endl;


    matchesSize = (int)allMatches.size();
    if(matchesSize == 0)
        return;

    if (out) *out << "grouping ..." << endl;

    Mat groupingMat((int)matchesSize, (int)matchesSize, CV_32F);
    groupingMat = Scalar(0);        
        
    /* grouping */
    for(int j = 0; j < matchesSize; ++j)
        for(int i = j + 1; i < matchesSize; ++i)        
        {
            const Match& mi = allMatches[i];
            const Match& mj = allMatches[j];

            if (mi.sceneInd == mj.sceneInd || mi.modelInd == mj.modelInd)
            {
                groupingMat.ptr<float>(i)[j] = float_max;
                groupingMat.ptr<float>(j)[i] = float_max;
                continue;
            }

            const Point3f& pointSceneI  = scene.getSpinVertex(mi.sceneInd);
            const Point3f& normalSceneI = scene.getSpinNormal(mi.sceneInd);
            
            const Point3f& pointModelI  = model.getSpinVertex(mi.modelInd);
            const Point3f& normalModelI = model.getSpinNormal(mi.modelInd);
            
            const Point3f& pointSceneJ  = scene.getSpinVertex(mj.sceneInd);
            const Point3f& normalSceneJ = scene.getSpinNormal(mj.sceneInd);
            
            const Point3f& pointModelJ  = model.getSpinVertex(mj.modelInd);
            const Point3f& normalModelJ = model.getSpinNormal(mj.modelInd);

            float wgc = groupingCreteria(pointSceneI, normalSceneI, pointModelI, normalModelI,
                                         pointSceneJ, normalSceneJ, pointModelJ, normalModelJ,
                                         model.gamma);   
            
            groupingMat.ptr<float>(i)[j] = wgc;
            groupingMat.ptr<float>(j)[i] = wgc;
        }

    group_t allMatchesInds;
    for(int i = 0; i < matchesSize; ++i)
        allMatchesInds.insert(i);
    
    vector<float> buf(matchesSize);
    float *buf_beg = &buf[0];
    vector<group_t> groups;
    
    for(int g = 0; g < matchesSize; ++g)
    {        
        if (out) if (g % 100 == 0) *out << "G = " << g << endl;

        group_t left = allMatchesInds;
        group_t group;
        
        left.erase(g);
        group.insert(g);
                        
        for(;;)
        {
            size_t left_size = left.size();
            if (left_size == 0)
                break;
                        
            std::transform(left.begin(), left.end(), buf_beg,  WgcHelper(group, groupingMat));
            size_t minInd = min_element(buf_beg, buf_beg + left_size) - buf_beg;
            
            if (buf[minInd] < model.T_GroupingCorespondances) /* can add corespondance to group */
            {
                iter pos = left.begin();
                advance(pos, minInd);
                
                group.insert(*pos);
                left.erase(pos);
            }
            else
                break;            
        }

        if (group.size() >= 4)
            groups.push_back(group);      
    }

    /* converting the data to final result */    
    for(size_t i = 0; i < groups.size(); ++i)
    {
        const group_t& group = groups[i];

        vector< Vec2i > outgrp;
        for(citer pos = group.begin(); pos != group.end(); ++pos)
        {
            const Match& m = allMatches[*pos];            
            outgrp.push_back(Vec2i(subset[m.modelInd], scene.subset[m.sceneInd]));
        }        
        result.push_back(outgrp);
    }    
}

cv::TickMeter::TickMeter() { reset(); }
int64 cv::TickMeter::getTimeTicks() const { return sumTime; }
double cv::TickMeter::getTimeMicro() const { return (double)getTimeTicks()/cvGetTickFrequency(); }
double cv::TickMeter::getTimeMilli() const { return getTimeMicro()*1e-3; }
double cv::TickMeter::getTimeSec()   const { return getTimeMilli()*1e-3; }    
int64 cv::TickMeter::getCounter() const { return counter; }
void  cv::TickMeter::reset() {startTime = 0; sumTime = 0; counter = 0; }

void cv::TickMeter::start(){ startTime = cvGetTickCount(); }
void cv::TickMeter::stop()
{
    int64 time = cvGetTickCount();
    if ( startTime == 0 )
        return;

    ++counter;

    sumTime += ( time - startTime );
    startTime = 0;
}

std::ostream& cv::operator<<(std::ostream& out, const TickMeter& tm){ return out << tm.getTimeSec() << "sec"; }
