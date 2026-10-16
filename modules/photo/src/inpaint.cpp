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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */

#include <type_traits>

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

using namespace cv;

#undef CV_MAT_ELEM_PTR_FAST
#define CV_MAT_ELEM_PTR_FAST( mat, row, col, pix_size )  \
     ((mat).data.ptr + (size_t)(mat).step*(row) + (pix_size)*(col))

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type round_cast(float val) {
   return cv::saturate_cast<T>(val);
}

template<typename T>
typename std::enable_if<!std::is_floating_point<T>::value, T>::type round_cast(float val) {
   return cv::saturate_cast<T>(val + 0.5);
}

inline float
min4( float a, float b, float c, float d )
{
    a = MIN(a,b);
    c = MIN(c,d);
    return MIN(a,c);
}

#define KNOWN  0  //known outside narrow band
#define BAND   1  //narrow band (known)
#define INSIDE 2  //unknown
#define CHANGE 3  //servise

static inline int inpaintSkipBlocksBelow(const uchar* p, int j, int n, uchar t)
{
#if CV_SIMD128
    const int nlanes = VTraits<v_uint8>::vlanes();
    for (; j <= n - nlanes && v_reduce_max(vx_load(p + j)) < t; j += nlanes) {}
#else
    CV_UNUSED(p); CV_UNUSED(n); CV_UNUSED(t);
#endif
    return j;
}

class CvPriorityQueueFloat
{
private:
    CvPriorityQueueFloat(const CvPriorityQueueFloat & ); // copy disabled
    CvPriorityQueueFloat& operator=(const CvPriorityQueueFloat &); // assign disabled

    struct Node
    {
        uint64 key;
        int i, j;
    };
    Node* storage;
    Node* nodes;
    size_t count, capacity;
    int next_order;

    static inline uint64 makeKey(float T, int order)
    {
        uint32_t bits;
        memcpy(&bits, &T, sizeof(bits));
        if ((bits << 1) == 0)
            bits = 0;
        bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        return ((uint64)bits << 32) | (uint32_t)order;
    }
    static inline float keyToT(uint64 key)
    {
        uint32_t bits = (uint32_t)(key >> 32);
        bits = (bits & 0x80000000u) ? (bits & 0x7fffffffu) : ~bits;
        float T;
        memcpy(&T, &bits, sizeof(T));
        return T;
    }

public:
    bool Add(const Mat &f) {
        for (int i = 0; i < f.rows; i++) {
            const uchar* fRow = f.ptr<uchar>(i);
            for (int j = inpaintSkipBlocksBelow(fRow, 0, f.cols, 1); j < f.cols; j = inpaintSkipBlocksBelow(fRow, j, f.cols, 1)) {
                for (int blockEnd = std::min(j + VTraits<v_uint8>::vlanes(), f.cols); j < blockEnd; j++) {
                    if (fRow[j] != 0 && !Push(i, j, 0))
                        return false;
                }
            }
        }
        return true;
    }

    bool Push(int i, int j, float T) {
        Node node;
        node.key = makeKey(T, next_order);
        node.i = i;
        node.j = j;
        ++next_order;
        if (count == capacity) {
            size_t newCapacity = capacity ? capacity * 2 : 256;
            Node* newStorage = (Node*)cv::fastMalloc((newCapacity + 3) * sizeof(Node));
            if (count)
                memcpy(newStorage + 3, nodes, count * sizeof(Node));
            cv::fastFree(storage);
            storage = newStorage;
            nodes = newStorage + 3;
            capacity = newCapacity;
        }
        size_t pos = count++;
        while (pos > 0) {
            size_t parent = (pos - 1) >> 2;
            if (nodes[parent].key < node.key)
                break;
            nodes[pos] = nodes[parent];
            pos = parent;
        }
        nodes[pos] = node;
        return true;
    }

    bool Pop(int *i, int *j) {
        if (count == 0) {
            return false;
        }
        *i = nodes[0].i;
        *j = nodes[0].j;
        const size_t size = --count;
        const Node last = nodes[size];
        if (size == 0)
            return true;
        size_t pos = 0;
        for (;;) {
            size_t child = 4 * pos + 1;
            if (child + 3 < size) {
                const uint64 key0 = nodes[child].key, key1 = nodes[child + 1].key, key2 = nodes[child + 2].key, key3 = nodes[child + 3].key;
                const size_t best01 = key1 < key0 ? child + 1 : child;
                const uint64 min01 = key1 < key0 ? key1 : key0;
                const size_t best23 = key3 < key2 ? child + 3 : child + 2;
                const uint64 min23 = key3 < key2 ? key3 : key2;
                const size_t best = min23 < min01 ? best23 : best01;
                const uint64 bestKey = min23 < min01 ? min23 : min01;
                if (last.key < bestKey)
                    break;
                nodes[pos] = nodes[best];
                pos = best;
                continue;
            }
            if (child >= size)
                break;
            size_t best = child;
            uint64 bestKey = nodes[child].key;
            for (size_t q = child + 1; q < size; q++) {
                if (nodes[q].key < bestKey) {
                    bestKey = nodes[q].key;
                    best = q;
                }
            }
            if (last.key < bestKey)
                break;
            nodes[pos] = nodes[best];
            pos = best;
        }
        nodes[pos] = last;
        return true;
    }

    bool Pop(int *i, int *j, float *T) {
        if (count == 0) {
            return false;
        }
        *T = keyToT(nodes[0].key);
        return Pop(i, j);
    }

    CvPriorityQueueFloat(void) : storage(), nodes(), count(), capacity(), next_order() {
    }
    ~CvPriorityQueueFloat() { cv::fastFree(storage); }
};

static inline float VectorScalMult(const cv::Point2f& v1, const cv::Point2f& v2)
{
   return v1.x*v2.x+v1.y*v2.y;
}

static inline float VectorLength(const cv::Point2f& v1)
{
    return v1.x*v1.x+v1.y*v1.y;
}

///////////////////////////////////////////////////////////////////////////////////////////
//HEAP::iterator Heap_Iterator;
//HEAP Heap;

#if !CV_SIMD128_64F
static float FastMarching_solve(int i1,int j1,int i2,int j2, const Mat &f, const Mat &t)
{
    double sol, a11, a22, m12;
    a11=t.at<float>(i1,j1);
    a22=t.at<float>(i2,j2);
    m12=MIN(a11,a22);

    if( f.at<uchar>(i1,j1) != INSIDE )
        if( f.at<uchar>(i2,j2) != INSIDE )
            if( fabs(a11-a22) >= 1.0 )
                sol = 1+m12;
            else
                sol = (a11+a22+sqrt((double)(2-(a11-a22)*(a11-a22))))*0.5;
        else
            sol = 1+a11;
    else if( f.at<uchar>(i2,j2) != INSIDE )
        sol = 1+a22;
    else
        sol = 1+m12;

    return (float)sol;
}
#endif

#if CV_SIMD128_64F
static inline v_float64x2 inpaintFastMarchingSolve2(const v_float64x2& a11, const v_float64x2& a22, const v_float64x2& known11, const v_float64x2& known22)
{
    const v_float64x2 one = v_setall_f64(1.), diff = v_sub(a11, a22), onePlusMin = v_add(one, v_min(a11, a22));
    const v_float64x2 sqrtSol = v_mul(v_add(v_add(a11, a22), v_sqrt(v_sub(v_setall_f64(2.), v_mul(diff, diff)))), v_setall_f64(0.5));
    return v_select(known11, v_select(known22, v_select(v_ge(v_abs(diff), one), onePlusMin, sqrtSol), v_add(one, a11)), v_select(known22, v_add(one, a22), onePlusMin));
}

static inline float inpaintFastMarchingMin4(const Mat &f, const Mat &t, int i, int j)
{
    const uchar* fRow = f.ptr<uchar>(i);
    const float* tRow = t.ptr<float>(i);
    const v_float64x2 tVert((double)t.ptr<float>(i-1)[j], (double)t.ptr<float>(i+1)[j]);
    const v_float64x2 knownVert = v_reinterpret_as_f64(v_uint64x2(f.ptr<uchar>(i-1)[j] != INSIDE ? ~(uint64)0 : 0, f.ptr<uchar>(i+1)[j] != INSIDE ? ~(uint64)0 : 0));
    const v_float64x2 knownLeft = v_reinterpret_as_f64(v_setall_u64(fRow[j-1] != INSIDE ? ~(uint64)0 : 0));
    const v_float64x2 knownRight = v_reinterpret_as_f64(v_setall_u64(fRow[j+1] != INSIDE ? ~(uint64)0 : 0));
    return v_reduce_min(v_cvt_f32(inpaintFastMarchingSolve2(tVert, v_setall_f64((double)tRow[j-1]), knownVert, knownLeft),
                                  inpaintFastMarchingSolve2(tVert, v_setall_f64((double)tRow[j+1]), knownVert, knownRight)));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////


static void
icvCalcFMM(Mat &f, Mat &t, CvPriorityQueueFloat *Heap, bool negate) {
   int i, j, ii = 0, jj = 0, q;
   float dist;

   while (Heap->Pop(&ii,&jj)) {

      unsigned known=(negate)?CHANGE:KNOWN;
      f.at<uchar>(ii,jj) = (uchar)known;

      for (q=0; q<4; q++) {
         i=0; j=0;
         if     (q==0) {i=ii-1; j=jj;}
         else if(q==1) {i=ii;   j=jj-1;}
         else if(q==2) {i=ii+1; j=jj;}
         else {i=ii;   j=jj+1;}
         if ((i<=0)||(j<=0)||(i>f.rows)||(j>f.cols)) continue;

         if (f.at<uchar>(i,j)==INSIDE) {
#if CV_SIMD128_64F
            dist = inpaintFastMarchingMin4(f, t, i, j);
#else
            dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                        FastMarching_solve(i+1,j,i,j-1,f,t),
                        FastMarching_solve(i-1,j,i,j+1,f,t),
                        FastMarching_solve(i+1,j,i,j+1,f,t));
#endif
            t.at<float>(i,j) = dist;
            f.at<uchar>(i,j) = BAND;
            Heap->Push(i,j,dist);
         }
      }
   }

   if (negate) {
      for (i=0; i<f.rows; i++) {
         uchar* fRow = f.ptr<uchar>(i);
         float* tRow = t.ptr<float>(i);
         for (j = inpaintSkipBlocksBelow(fRow, 0, f.cols, CHANGE); j < f.cols; j = inpaintSkipBlocksBelow(fRow, j, f.cols, CHANGE)) {
            for (int blockEnd = std::min(j + VTraits<v_uint8>::vlanes(), f.cols); j < blockEnd; j++) {
               if (fRow[j] == CHANGE) {
                  fRow[j] = KNOWN;
                  tRow[j] = -tRow[j];
               }
            }
         }
      }
   }
}

#if CV_SIMD128

static inline v_int32x4 inpaintLoadPixel3(const uchar* p, const uchar* pixelLoadEnd)
{
    return p < pixelLoadEnd ? v_reinterpret_as_s32(v_load_expand_q(p)) : v_int32x4(p[0], p[1], p[2], 0);
}

struct InpaintDiskTable
{
    int range, side;
    cv::AutoBuffer<int> halfWidth;
    cv::AutoBuffer<float> dst;

    explicit InpaintDiskTable(int r) : range(r), side(2 * r + 1), halfWidth(2 * r + 1), dst((2 * r + 1) * (2 * r + 1))
    {
        for (int dk = -r; dk <= r; dk++)
        {
            int rowHalfWidth = 0;
            while ((rowHalfWidth + 1) * (rowHalfWidth + 1) + dk * dk <= r * r)
                rowHalfWidth++;
            halfWidth[dk + r] = rowHalfWidth;
        }
    }
    float* dstRow(int dk) { return dst.data() + (dk + range) * side + range; }
};

static float inpaintDirThreshold()
{
    float result = (float)0.01;
    while ((double)result > 0.01)
        result = std::nextafter(result, 0.f);
    while ((double)std::nextafter(result, 1.f) <= 0.01)
        result = std::nextafter(result, 1.f);
    return result;
}

static inline cv::Point2f inpaintGradientT(const Mat &f, const Mat &t, int i, int j)
{
    cv::Point2f gradT;
    if (f.at<uchar>(i,j+1)!=INSIDE) {
        if (f.at<uchar>(i,j-1)!=INSIDE) {
            gradT.x=(float)((t.at<float>(i,j+1)-t.at<float>(i,j-1)))*0.5f;
        } else {
            gradT.x=(float)((t.at<float>(i,j+1)-t.at<float>(i,j)));
        }
    } else {
        if (f.at<uchar>(i,j-1)!=INSIDE) {
            gradT.x=(float)((t.at<float>(i,j)-t.at<float>(i,j-1)));
        } else {
            gradT.x=0;
        }
    }
    if (f.at<uchar>(i+1,j)!=INSIDE) {
        if (f.at<uchar>(i-1,j)!=INSIDE) {
            gradT.y=(float)((t.at<float>(i+1,j)-t.at<float>(i-1,j)))*0.5f;
        } else {
            gradT.y=(float)((t.at<float>(i+1,j)-t.at<float>(i,j)));
        }
    } else {
        if (f.at<uchar>(i-1,j)!=INSIDE) {
            gradT.y=(float)((t.at<float>(i,j)-t.at<float>(i-1,j)));
        } else {
            gradT.y=0;
        }
    }
    return gradT;
}

static void inpaintTeleaRowWeights(const float* tRow, float tij, const float* dstRow, int j, int lbeg, int lend,
                             float ry, const cv::Point2f& gradT, float dirThreshold, float* w)
{
    const v_int32x4 laneIdx(0, 1, 2, 3);
    const v_float32x4 vtij = v_setall_f32(tij), vone = v_setall_f32(1.f);
    const v_float32x4 vRyGy = v_mul(v_setall_f32(ry), v_setall_f32(gradT.y));
    const v_float32x4 vgx = v_setall_f32(gradT.x), vthr = v_setall_f32(dirThreshold), vsmall = v_setall_f32(0.000001f);
#if CV_SIMD128_64F
    const v_float64x2 vone64 = v_setall_f64(1.);
#endif
    const int nlanes = VTraits<v_float32x4>::vlanes();
    int l = lbeg;
    for (; l <= lend - nlanes + 1; l += nlanes)
    {
        v_float32x4 rx = v_cvt_f32(v_sub(v_setall_s32(j - l), laneIdx));
        v_float32x4 b = v_add(v_abs(v_sub(v_load(tRow + l), vtij)), vone);
#if CV_SIMD128_64F
        v_float32x4 lev = v_cvt_f32(v_div(vone64, v_cvt_f64(b)), v_div(vone64, v_cvt_f64_high(b)));
#else
        float bArr[4], levArr[4];
        v_store(bArr, b);
        for (int lane = 0; lane < 4; lane++)
            levArr[lane] = (float)(1./(double)bArr[lane]);
        v_float32x4 lev = v_load(levArr);
#endif
        v_float32x4 dir = v_add(v_mul(rx, vgx), vRyGy);
        dir = v_select(v_le(v_abs(dir), vthr), vsmall, dir);
        v_float32x4 dst = v_load(dstRow + (l - j));
        v_store(w + (l - lbeg), v_abs(v_mul(v_mul(dst, lev), dir)));
    }
    for (; l <= lend; l++)
    {
        cv::Point2f r;
        r.y = ry;
        r.x = (float)(j-l);
        float lev = (float)(1./(1+fabs(tRow[l]-tij)));
        float dir = VectorScalMult(r,gradT);
        if (fabs(dir)<=0.01) dir=0.000001f;
        w[l - lbeg] = (float)fabs(dstRow[l - j]*lev*dir);
    }
}

template <typename data_type>
static void
icvTeleaInpaintFMM_simd(Mat &f, Mat &t, Mat &out, int range, CvPriorityQueueFloat *Heap ) {
   int i = 0, j = 0, ii = 0, jj = 0, k, l, q;
   float dist;
   const bool c3 = out.channels()==3;
   const float dirThreshold = inpaintDirThreshold();
   const uchar* pixelLoadEnd = c3 && !out.empty() ? out.dataend - 3 : out.dataend;

   InpaintDiskTable diskTab(range);
   for (int dk = -range; dk <= range; dk++) {
      float* drow = diskTab.dstRow(dk);
      for (int dl = -range; dl <= range; dl++) {
         if (dk == 0 && dl == 0) { drow[dl] = 0.f; continue; }
         cv::Point2f r;
         r.y = (float)(-dk);
         r.x = (float)(-dl);
         if (c3)
            drow[dl] = (float)(1./(VectorLength(r)*sqrt((double)VectorLength(r))));
         else
            drow[dl] = (float)(1./(VectorLength(r)*sqrt(VectorLength(r))));
      }
   }
   cv::AutoBuffer<float> _weightBuf(diskTab.side + 4);
   float* weightBuf = _weightBuf.data();

   while (Heap->Pop(&ii,&jj)) {

      f.at<uchar>(ii,jj) = KNOWN;
      for(q=0; q<4; q++) {
         if     (q==0) {i=ii-1; j=jj;}
         else if(q==1) {i=ii;   j=jj-1;}
         else if(q==2) {i=ii+1; j=jj;}
         else if(q==3) {i=ii;   j=jj+1;}
         if ((i<=0)||(j<=0)||(i>t.rows-1)||(j>t.cols-1)) continue;

         if (f.at<uchar>(i,j)==INSIDE) {
#if CV_SIMD128_64F
            dist = inpaintFastMarchingMin4(f, t, i, j);
#else
            dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                        FastMarching_solve(i+1,j,i,j-1,f,t),
                        FastMarching_solve(i-1,j,i,j+1,f,t),
                        FastMarching_solve(i+1,j,i,j+1,f,t));
#endif
            t.at<float>(i,j) = dist;

            const cv::Point2f gradT = inpaintGradientT(f, t, i, j);
            const float tij = t.at<float>(i,j);
            const int kbeg = std::max(i-range, 1), kend = std::min(i+range, t.rows-2);

            if (c3) {
               v_float32x4 vIa = v_setall_f32(0.f), vJx = v_setall_f32(0.f), vJy = v_setall_f32(0.f);
               v_float32x4 vs = v_setall_f32(1.0e-20f);
               for (k=kbeg; k<=kend; k++) {
                  int km=k-1+(k==1),kp=k-1-(k==t.rows-2);
                  int rowHalfWidth = diskTab.halfWidth[k-i+range];
                  int lbeg = std::max(j-rowHalfWidth, 1), lend = std::min(j+rowHalfWidth, t.cols-2);
                  if (lbeg > lend) continue;
                  const float ry = (float)(i-k);
                  inpaintTeleaRowWeights(t.ptr<float>(k), tij, diskTab.dstRow(k-i), j, lbeg, lend, ry, gradT, dirThreshold, weightBuf);
                  const uchar* fk = f.ptr<uchar>(k);
                  const uchar* fkm1 = f.ptr<uchar>(k-1);
                  const uchar* fkp1 = f.ptr<uchar>(k+1);
                  const uchar* o_km = out.ptr<uchar>(km);
                  const uchar* o_kp = out.ptr<uchar>(kp);
                  const uchar* o_kp1 = out.ptr<uchar>(kp+1);
                  const uchar* o_km1 = out.ptr<uchar>(km-1);
                  const uchar* o_k1 = out.ptr<uchar>(k-1);
                  const v_float32x4 vry = v_setall_f32(ry);
                  for (l=lbeg; l<=lend; l++) {
                     if (fk[l]==INSIDE) continue;
                     int lm=l-1+(l==1),lp=l-1-(l==t.cols-2);
                     const v_float32x4 vw = v_setall_f32(weightBuf[l-lbeg]);
                     const v_float32x4 vrx = v_setall_f32((float)(j-l));
                     v_float32x4 gx, gy;
                     if (fk[l+1]!=INSIDE) {
                        if (fk[l-1]!=INSIDE) {
                           gx = v_mul(v_cvt_f32(v_sub(inpaintLoadPixel3(o_km + 3*(lp+1), pixelLoadEnd),
                                                      inpaintLoadPixel3(o_km + 3*(lm-1), pixelLoadEnd))), v_setall_f32(2.0f));
                        } else {
                           gx = v_cvt_f32(v_sub(inpaintLoadPixel3(o_km + 3*(lp+1), pixelLoadEnd),
                                                inpaintLoadPixel3(o_km + 3*lm, pixelLoadEnd)));
                        }
                     } else {
                        if (fk[l-1]!=INSIDE) {
                           gx = v_cvt_f32(v_sub(inpaintLoadPixel3(o_km + 3*lp, pixelLoadEnd),
                                                inpaintLoadPixel3(o_km + 3*(lm-1), pixelLoadEnd)));
                        } else {
                           gx = v_setall_f32(0.f);
                        }
                     }
                     if (fkp1[l]!=INSIDE) {
                        if (fkm1[l]!=INSIDE) {
                           gy = v_mul(v_cvt_f32(v_sub(inpaintLoadPixel3(o_kp1 + 3*lm, pixelLoadEnd),
                                                      inpaintLoadPixel3(o_km1 + 3*lm, pixelLoadEnd))), v_setall_f32(2.0f));
                        } else {
                           gy = v_cvt_f32(v_sub(inpaintLoadPixel3(o_kp1 + 3*lm, pixelLoadEnd),
                                                inpaintLoadPixel3(o_km + 3*lm, pixelLoadEnd)));
                        }
                     } else {
                        if (fkm1[l]!=INSIDE) {
                           gy = v_cvt_f32(v_sub(inpaintLoadPixel3(o_kp + 3*lm, pixelLoadEnd),
                                                inpaintLoadPixel3(o_km1 + 3*lm, pixelLoadEnd)));
                        } else {
                           gy = v_setall_f32(0.f);
                        }
                     }
                     vIa = v_add(vIa, v_mul(vw, v_cvt_f32(inpaintLoadPixel3(o_k1 + 3*(l-1), pixelLoadEnd))));
                     vJx = v_sub(vJx, v_mul(vw, v_mul(gx, vrx)));
                     vJy = v_sub(vJy, v_mul(vw, v_mul(gy, vry)));
                     vs = v_add(vs, vw);
                  }
               }
               float Ia[4], Jx[4], Jy[4], s[4];
               v_store(Ia, vIa); v_store(Jx, vJx); v_store(Jy, vJy); v_store(s, vs);
               for (int color=0; color<=2; color++) {
                  float sat = (float)(Ia[color]/s[color]+(Jx[color]+Jy[color])/(sqrt(Jx[color]*Jx[color]+Jy[color]*Jy[color])+1.0e-20f));
                  out.at<Vec<uchar, 3> >(i-1,j-1)[color] = round_cast<uchar>(sat);
               }
            } else {
               cv::Point2f gradI,r;
               float Ia=0,Jx=0,Jy=0,s=1.0e-20f,w,sat;
               for (k=kbeg; k<=kend; k++) {
                  int km=k-1+(k==1),kp=k-1-(k==t.rows-2);
                  int rowHalfWidth = diskTab.halfWidth[k-i+range];
                  int lbeg = std::max(j-rowHalfWidth, 1), lend = std::min(j+rowHalfWidth, t.cols-2);
                  if (lbeg > lend) continue;
                  const float ry = (float)(i-k);
                  inpaintTeleaRowWeights(t.ptr<float>(k), tij, diskTab.dstRow(k-i), j, lbeg, lend, ry, gradT, dirThreshold, weightBuf);
                  const uchar* fk = f.ptr<uchar>(k);
                  const uchar* fkm1 = f.ptr<uchar>(k-1);
                  const uchar* fkp1 = f.ptr<uchar>(k+1);
                  const data_type* o_km = out.ptr<data_type>(km);
                  const data_type* o_kp = out.ptr<data_type>(kp);
                  const data_type* o_kp1 = out.ptr<data_type>(kp+1);
                  const data_type* o_km1 = out.ptr<data_type>(km-1);
                  const data_type* o_k1 = out.ptr<data_type>(k-1);
                  for (l=lbeg; l<=lend; l++) {
                     if (fk[l]==INSIDE) continue;
                     int lm=l-1+(l==1),lp=l-1-(l==t.cols-2);
                     r.y = ry;
                     r.x = (float)(j-l);
                     w = weightBuf[l-lbeg];

                     if (fk[l+1]!=INSIDE) {
                        if (fk[l-1]!=INSIDE) {
                           gradI.x=(float)((o_km[lp+1]-o_km[lm-1]))*2.0f;
                        } else {
                           gradI.x=(float)((o_km[lp+1]-o_km[lm]));
                        }
                     } else {
                        if (fk[l-1]!=INSIDE) {
                           gradI.x=(float)((o_km[lp]-o_km[lm-1]));
                        } else {
                           gradI.x=0;
                        }
                     }
                     if (fkp1[l]!=INSIDE) {
                        if (fkm1[l]!=INSIDE) {
                           gradI.y=(float)((o_kp1[lm]-o_km1[lm]))*2.0f;
                        } else {
                           gradI.y=(float)((o_kp1[lm]-o_km[lm]));
                        }
                     } else {
                        if (fkm1[l]!=INSIDE) {
                           gradI.y=(float)((o_kp[lm]-o_km1[lm]));
                        } else {
                           gradI.y=0;
                        }
                     }
                     Ia += (float)w * (float)(o_k1[l-1]);
                     Jx -= (float)w * (float)(gradI.x*r.x);
                     Jy -= (float)w * (float)(gradI.y*r.y);
                     s  += w;
                  }
               }
               sat = (float)(Ia/s+(Jx+Jy)/(sqrt(Jx*Jx+Jy*Jy)+1.0e-20f));
               out.at<data_type>(i-1,j-1) = round_cast<data_type>(sat);
            }

            f.at<uchar>(i,j) = BAND;
            Heap->Push(i,j,dist);
         }
      }
   }
}

static inline v_int32x4 inpaintNSLoad4(const uchar* p) { return v_reinterpret_as_s32(v_load_expand_q(p)); }
static inline v_int32x4 inpaintNSLoad4(const ushort* p) { return v_reinterpret_as_s32(v_load_expand(p)); }
static inline v_float32x4 inpaintNSLoad4(const float* p) { return v_load(p); }
static inline v_int32x4 inpaintNSAbsDiff(const v_int32x4& a, const v_int32x4& b) { return v_reinterpret_as_s32(v_absdiff(a, b)); }
static inline v_float32x4 inpaintNSAbsDiff(const v_float32x4& a, const v_float32x4& b) { return v_absdiff(a, b); }
static inline v_float32x4 inpaintNSSumToFloat(const v_int32x4& a, const v_int32x4& b) { return v_cvt_f32(v_add(a, b)); }
static inline v_float32x4 inpaintNSSumToFloat(const v_float32x4& a, const v_float32x4& b) { return v_add(a, b); }
static inline v_float32x4 inpaintNSToFloat(const v_int32x4& a) { return v_cvt_f32(a); }
static inline v_float32x4 inpaintNSToFloat(const v_float32x4& a) { return a; }
static inline v_float32x4 inpaintNSKnownMask4(const uchar* p)
{
    return v_reinterpret_as_f32(v_ne(v_load_expand_q(p), v_setall_u32(INSIDE)));
}

template <typename data_type>
static int inpaintNSRowWeights(const uchar* fk, const uchar* fkm1, const uchar* fkp1,
                         const data_type* o_k2, const data_type* o_k1, const data_type* o_k,
                         const float* drow, int i, int k, int j, int lbeg, int lend, float dirThreshold, float* w)
{
    const v_int32x4 laneIdx(0, 1, 2, 3);
    const float ry = (float)(i-k);
    const v_float32x4 vry = v_setall_f32(ry), vRySq = v_mul(vry, vry);
    const v_float32x4 vzero = v_setall_f32(0.f), vthr = v_setall_f32(dirThreshold), vsmall = v_setall_f32(0.000001f);
    const int nlanes = VTraits<v_float32x4>::vlanes();
    int l = lbeg;
    for (; l <= lend - nlanes + 1; l += nlanes)
    {
        const v_float32x4 kp1 = inpaintNSKnownMask4(fkp1 + l), km1 = inpaintNSKnownMask4(fkm1 + l);
        const v_float32x4 lp1 = inpaintNSKnownMask4(fk + l + 1), lm1 = inpaintNSKnownMask4(fk + l - 1);
        const auto A = inpaintNSLoad4(o_k + l - 1);
        const auto B = inpaintNSLoad4(o_k1 + l - 1);
        const auto C = inpaintNSLoad4(o_k2 + l - 1);
        const auto absAB = inpaintNSAbsDiff(A, B), absBC = inpaintNSAbsDiff(B, C);
        v_float32x4 gx = v_select(kp1,
                                   v_select(km1, inpaintNSSumToFloat(absAB, absBC), v_mul(inpaintNSToFloat(absAB), v_setall_f32(2.0f))),
                                   v_select(km1, v_mul(inpaintNSToFloat(absBC), v_setall_f32(2.0f)), vzero));
        const auto D = inpaintNSLoad4(o_k1 + l);
        const auto F = inpaintNSLoad4(o_k1 + l - 2);
        const auto absDB = inpaintNSAbsDiff(D, B), absBF = inpaintNSAbsDiff(B, F);
        const v_float32x4 gy = v_select(lp1,
                                         v_select(lm1, inpaintNSSumToFloat(absDB, absBF), v_mul(inpaintNSToFloat(absDB), v_setall_f32(2.0f))),
                                         v_select(lm1, v_mul(inpaintNSToFloat(absBF), v_setall_f32(2.0f)), vzero));
        gx = v_xor(gx, v_setall_f32(-0.f));
        const v_float32x4 rx = v_cvt_f32(v_sub(v_setall_s32(j - l), laneIdx));
        v_float32x4 dir = v_add(v_mul(rx, gx), v_mul(vry, gy));
        const v_float32x4 rLen2 = v_add(v_mul(rx, rx), vRySq);
        const v_float32x4 gradLen2 = v_add(v_mul(gx, gx), v_mul(gy, gy));
        const v_float32x4 smallMask = v_le(v_abs(dir), vthr);
        const v_float32x4 safeDenom = v_select(smallMask, v_setall_f32(1.f), v_sqrt(v_mul(rLen2, gradLen2)));
        const v_float32x4 dirNorm = v_abs(v_div(dir, safeDenom));
        dir = v_select(smallMask, vsmall, dirNorm);
        v_store(w + (l - lbeg), v_mul(v_load(drow + (l - j)), dir));
    }
    return l;
}

template <typename data_type>
static void
icvNSInpaintFMM_simd(Mat &f, Mat &t, Mat &out, int range, CvPriorityQueueFloat *Heap) {
   int i = 0, j = 0, ii = 0, jj = 0, k, l, q;
   float dist;
   const bool c3 = out.channels()==3;
   const float dirThreshold = inpaintDirThreshold();
   const uchar* pixelLoadEnd = c3 && !out.empty() ? out.dataend - 3 : out.dataend;

   InpaintDiskTable diskTab(range);
   for (int dk = -range; dk <= range; dk++) {
      float* drow = diskTab.dstRow(dk);
      for (int dl = -range; dl <= range; dl++) {
         cv::Point2f r;
         if (c3) { r.y=(float)(dk); r.x=(float)(dl); }
         else    { r.y=(float)(-dk); r.x=(float)(-dl); }
         drow[dl] = 1/(VectorLength(r)*VectorLength(r)+1);
      }
   }
   cv::AutoBuffer<float> _weightBuf(diskTab.side + 4);
   float* weightBuf = _weightBuf.data();

   while (Heap->Pop(&ii,&jj)) {

      f.at<uchar>(ii,jj) = KNOWN;
      for(q=0; q<4; q++) {
         if     (q==0) {i=ii-1; j=jj;}
         else if(q==1) {i=ii;   j=jj-1;}
         else if(q==2) {i=ii+1; j=jj;}
         else if(q==3) {i=ii;   j=jj+1;}
         if ((i<=0)||(j<=0)||(i>t.rows-1)||(j>t.cols-1)) continue;

         if (f.at<uchar>(i,j)==INSIDE) {
#if CV_SIMD128_64F
            dist = inpaintFastMarchingMin4(f, t, i, j);
#else
            dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                        FastMarching_solve(i+1,j,i,j-1,f,t),
                        FastMarching_solve(i-1,j,i,j+1,f,t),
                        FastMarching_solve(i+1,j,i,j+1,f,t));
#endif
            t.at<float>(i,j) = dist;

            const int kbeg = std::max(i-range, 1), kend = std::min(i+range, t.rows-2);

            if (c3) {
               const v_float32x4 vthr = v_setall_f32(dirThreshold), vsmall = v_setall_f32(0.000001f);
               v_float32x4 vIa = v_setall_f32(0.f), vs = v_setall_f32(1.0e-20f);
               for (k=kbeg; k<=kend; k++) {
                  int km=k-1+(k==1),kp=k-1-(k==f.rows-2);
                  int rowHalfWidth = diskTab.halfWidth[k-i+range];
                  int lbeg = std::max(j-rowHalfWidth, 1), lend = std::min(j+rowHalfWidth, f.cols-2);
                  const float* drow = diskTab.dstRow(k-i);
                  const uchar* fk = f.ptr<uchar>(k);
                  const uchar* fkm1 = f.ptr<uchar>(k-1);
                  const uchar* fkp1 = f.ptr<uchar>(k+1);
                  const uchar* o_km = out.ptr<uchar>(km);
                  const uchar* o_kp = out.ptr<uchar>(kp);
                  const uchar* o_kp1 = out.ptr<uchar>(kp+1);
                  const uchar* o_km1 = out.ptr<uchar>(km-1);
                  const uchar* o_k1 = out.ptr<uchar>(k-1);
                  const float ry = (float)(k-i);
                  for (l=lbeg; l<=lend; l++) {
                     if (fk[l]==INSIDE) continue;
                     int lm=l-1+(l==1),lp=l-1-(l==f.cols-2);
                     cv::Point2f r;
                     r.y=ry;
                     r.x=(float)(l-j);
                     const v_float32x4 vdst = v_setall_f32(drow[l-j]);
                     v_float32x4 gx, gy;
                     if (fkp1[l]!=INSIDE) {
                        if (fkm1[l]!=INSIDE) {
                           const v_int32x4 b = inpaintLoadPixel3(o_kp + 3*lm, pixelLoadEnd);
                           gx = v_cvt_f32(v_reinterpret_as_s32(v_add(
                                    v_absdiff(inpaintLoadPixel3(o_kp1 + 3*lm, pixelLoadEnd), b),
                                    v_absdiff(b, inpaintLoadPixel3(o_km1 + 3*lm, pixelLoadEnd)))));
                        } else {
                           gx = v_mul(v_cvt_f32(v_reinterpret_as_s32(v_absdiff(inpaintLoadPixel3(o_kp1 + 3*lm, pixelLoadEnd),
                                                                               inpaintLoadPixel3(o_kp + 3*lm, pixelLoadEnd)))), v_setall_f32(2.0f));
                        }
                     } else {
                        if (fkm1[l]!=INSIDE) {
                           gx = v_mul(v_cvt_f32(v_reinterpret_as_s32(v_absdiff(inpaintLoadPixel3(o_kp + 3*lm, pixelLoadEnd),
                                                                               inpaintLoadPixel3(o_km1 + 3*lm, pixelLoadEnd)))), v_setall_f32(2.0f));
                        } else {
                           gx = v_setall_f32(0.f);
                        }
                     }
                     if (fk[l+1]!=INSIDE) {
                        if (fk[l-1]!=INSIDE) {
                           const v_int32x4 b = inpaintLoadPixel3(o_km + 3*lm, pixelLoadEnd);
                           gy = v_cvt_f32(v_reinterpret_as_s32(v_add(
                                    v_absdiff(inpaintLoadPixel3(o_km + 3*(lp+1), pixelLoadEnd), b),
                                    v_absdiff(b, inpaintLoadPixel3(o_km + 3*(lm-1), pixelLoadEnd)))));
                        } else {
                           gy = v_mul(v_cvt_f32(v_reinterpret_as_s32(v_absdiff(inpaintLoadPixel3(o_km + 3*(lp+1), pixelLoadEnd),
                                                                               inpaintLoadPixel3(o_km + 3*lm, pixelLoadEnd)))), v_setall_f32(2.0f));
                        }
                     } else {
                        if (fk[l-1]!=INSIDE) {
                           gy = v_mul(v_cvt_f32(v_reinterpret_as_s32(v_absdiff(inpaintLoadPixel3(o_km + 3*lm, pixelLoadEnd),
                                                                               inpaintLoadPixel3(o_km + 3*(lm-1), pixelLoadEnd)))), v_setall_f32(2.0f));
                        } else {
                           gy = v_setall_f32(0.f);
                        }
                     }
                     gx = v_xor(gx, v_setall_f32(-0.f));
                     v_float32x4 dir = v_add(v_mul(gx, v_setall_f32(r.x)), v_mul(gy, v_setall_f32(r.y)));
                     v_float32x4 gradLen2 = v_add(v_mul(gx, gx), v_mul(gy, gy));
                     v_float32x4 denom = v_sqrt(v_mul(gradLen2, v_setall_f32(VectorLength(r))));
                     v_float32x4 smallMask = v_le(v_abs(dir), vthr);
                     v_float32x4 safeDenom = v_select(smallMask, v_setall_f32(1.f), denom);
                     v_float32x4 dirNorm = v_abs(v_div(dir, safeDenom));
                     dir = v_select(smallMask, vsmall, dirNorm);
                     const v_float32x4 vw = v_mul(vdst, dir);
                     vIa = v_add(vIa, v_mul(vw, v_cvt_f32(inpaintLoadPixel3(o_k1 + 3*(l-1), pixelLoadEnd))));
                     vs = v_add(vs, vw);
                  }
               }
               float Ia[4], s[4];
               v_store(Ia, vIa); v_store(s, vs);
               for (int color=0; color<=2; color++) {
                  out.at<Vec<uchar, 3> >(i-1,j-1)[color] = cv::saturate_cast<uchar>((double)Ia[color]/s[color]);
               }
            } else {
               cv::Point2f gradI,r;
               float Ia=0,s=1.0e-20f,w,dst,dir;
               for (k=kbeg; k<=kend; k++) {
                  int km=k-1+(k==1),kp=k-1-(k==t.rows-2);
                  int rowHalfWidth = diskTab.halfWidth[k-i+range];
                  int lbeg = std::max(j-rowHalfWidth, 1), lend = std::min(j+rowHalfWidth, t.cols-2);
                  const float* drow = diskTab.dstRow(k-i);
                  const uchar* fk = f.ptr<uchar>(k);
                  const uchar* fkm1 = f.ptr<uchar>(k-1);
                  const uchar* fkp1 = f.ptr<uchar>(k+1);
                  const data_type* o_km = out.ptr<data_type>(km);
                  const data_type* o_kp = out.ptr<data_type>(kp);
                  const data_type* o_kp1 = out.ptr<data_type>(kp+1);
                  const data_type* o_km1 = out.ptr<data_type>(km-1);
                  const data_type* o_k1 = out.ptr<data_type>(k-1);
                  int vecBegin = std::max(lbeg, 2), vecEnd = vecBegin;
                  if (k != 1 && k != t.rows-2 && vecBegin <= std::min(lend, t.cols-3))
                     vecEnd = inpaintNSRowWeights<data_type>(fk, fkm1, fkp1, out.ptr<data_type>(k-2), o_k1, out.ptr<data_type>(k),
                                                   drow, i, k, j, vecBegin, std::min(lend, t.cols-3), dirThreshold, weightBuf);
                  for (l=lbeg; l<=lend; l++) {
                     if (fk[l]==INSIDE) continue;
                     if (l >= vecBegin && l < vecEnd) {
                        w = weightBuf[l-vecBegin];
                        Ia += (float)w * (float)(o_k1[l-1]);
                        s  += w;
                        continue;
                     }
                     int lm=l-1+(l==1),lp=l-1-(l==t.cols-2);
                     r.y=(float)(i-k);
                     r.x=(float)(j-l);

                     dst = drow[l-j];

                     if (fkp1[l]!=INSIDE) {
                        if (fkm1[l]!=INSIDE) {
                           gradI.x=(float)(std::abs(o_kp1[lm]-o_kp[lm])+
                                           std::abs(o_kp[lm]-o_km1[lm]));
                        } else {
                           gradI.x=(float)(std::abs(o_kp1[lm]-o_kp[lm]))*2.0f;
                        }
                     } else {
                        if (fkm1[l]!=INSIDE) {
                           gradI.x=(float)(std::abs(o_kp[lm]-o_km1[lm]))*2.0f;
                        } else {
                           gradI.x=0;
                        }
                     }
                     if (fk[l+1]!=INSIDE) {
                        if (fk[l-1]!=INSIDE) {
                           gradI.y=(float)(std::abs(o_km[lp+1]-o_km[lm])+
                                           std::abs(o_km[lm]-o_km[lm-1]));
                        } else {
                           gradI.y=(float)(std::abs(o_km[lp+1]-o_km[lm]))*2.0f;
                        }
                     } else {
                        if (fk[l-1]!=INSIDE) {
                           gradI.y=(float)(std::abs(o_km[lm]-o_km[lm-1]))*2.0f;
                        } else {
                           gradI.y=0;
                        }
                     }

                     gradI.x=-gradI.x;
                     dir=VectorScalMult(r,gradI);

                     if (fabs(dir)<=0.01) {
                        dir=0.000001f;
                     } else {
                        dir = (float)fabs(VectorScalMult(r,gradI)/sqrt(VectorLength(r)*VectorLength(gradI)));
                     }
                     w = dst*dir;
                     Ia += (float)w * (float)(o_k1[l-1]);
                     s  += w;
                  }
               }
               out.at<data_type>(i-1,j-1) = cv::saturate_cast<data_type>((double)Ia/s);
            }

            f.at<uchar>(i,j) = BAND;
            Heap->Push(i,j,dist);
         }
      }
   }
}
#endif

template <typename data_type>
static void
icvTeleaInpaintFMM(Mat &f, Mat &t, Mat &out, int range, CvPriorityQueueFloat *Heap ) {
#if CV_SIMD128
   icvTeleaInpaintFMM_simd<data_type>(f, t, out, range, Heap);
#else
   int i = 0, j = 0, ii = 0, jj = 0, k, l, q, color = 0;
   float dist;

   if (out.channels()==3) {
      typedef Vec<uchar, 3> PixelT;

      while (Heap->Pop(&ii,&jj)) {

         f.at<uchar>(ii,jj) = KNOWN;
         for(q=0; q<4; q++) {
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>t.rows-1)||(j>t.cols-1)) continue;

            if (f.at<uchar>(i,j)==INSIDE) {
               dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
               t.at<float>(i,j) = dist;

               cv::Point2f gradT[3];
               for (color=0; color<=2; color++) {
                  if (f.at<uchar>(i,j+1)!=INSIDE) {
                     if (f.at<uchar>(i,j-1)!=INSIDE) {
                        gradT[color].x=(float)((t.at<float>(i,j+1)-t.at<float>(i,j-1)))*0.5f;
                     } else {
                        gradT[color].x=(float)((t.at<float>(i,j+1)-t.at<float>(i,j)));
                     }
                  } else {
                     if (f.at<uchar>(i,j-1)!=INSIDE) {
                        gradT[color].x=(float)((t.at<float>(i,j)-t.at<float>(i,j-1)));
                     } else {
                        gradT[color].x=0;
                     }
                  }
                  if (f.at<uchar>(i+1,j)!=INSIDE) {
                     if (f.at<uchar>(i-1,j)!=INSIDE) {
                        gradT[color].y=(float)((t.at<float>(i+1,j)-t.at<float>(i-1,j)))*0.5f;
                     } else {
                        gradT[color].y=(float)((t.at<float>(i+1,j)-t.at<float>(i,j)));
                     }
                  } else {
                     if (f.at<uchar>(i-1,j)!=INSIDE) {
                        gradT[color].y=(float)((t.at<float>(i,j)-t.at<float>(i-1,j)));
                     } else {
                        gradT[color].y=0;
                     }
                  }
               }

               cv::Point2f gradI,r;
               float Jx[3] = {0,0,0};
               float Jy[3] = {0,0,0};
               float Ia[3] = {0,0,0};
               float s[3] = {1.0e-20f,1.0e-20f,1.0e-20f};
               float w,dst,lev,dir,sat;

               for (k=i-range; k<=i+range; k++) {
                  int km=k-1+(k==1),kp=k-1-(k==t.rows-2);
                  for (l=j-range; l<=j+range; l++) {
                     int lm=l-1+(l==1),lp=l-1-(l==t.cols-2);
                     if (k>0&&l>0&&k<t.rows-1&&l<t.cols-1) {
                        if ((f.at<uchar>(k,l)!=INSIDE)&&
                            ((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                           for (color=0; color<=2; color++) {
                              r.y     = (float)(i-k);
                              r.x     = (float)(j-l);

                              dst = (float)(1./(VectorLength(r)*sqrt((double)VectorLength(r))));
                              lev = (float)(1./(1+fabs(t.at<float>(k,l)-t.at<float>(i,j))));

                              dir=VectorScalMult(r,gradT[color]);
                              if (fabs(dir)<=0.01) dir=0.000001f;
                              w = (float)fabs(dst*lev*dir);

                              if (f.at<uchar>(k,l+1)!=INSIDE) {
                                 if (f.at<uchar>(k,l-1)!=INSIDE) {
                                    gradI.x=(float)((out.at<PixelT>(km,lp+1)[color]-out.at<PixelT>(km,lm-1)[color]))*2.0f;
                                 } else {
                                    gradI.x=(float)((out.at<PixelT>(km,lp+1)[color]-out.at<PixelT>(km,lm)[color]));
                                 }
                              } else {
                                 if (f.at<uchar>(k,l-1)!=INSIDE) {
                                    gradI.x=(float)((out.at<PixelT>(km,lp)[color]-out.at<PixelT>(km,lm-1)[color]));
                                 } else {
                                    gradI.x=0;
                                 }
                              }
                              if (f.at<uchar>(k+1,l)!=INSIDE) {
                                 if (f.at<uchar>(k-1,l)!=INSIDE) {
                                    gradI.y=(float)((out.at<PixelT>(kp+1,lm)[color]-out.at<PixelT>(km-1,lm)[color]))*2.0f;
                                 } else {
                                    gradI.y=(float)((out.at<PixelT>(kp+1,lm)[color]-out.at<PixelT>(km,lm)[color]));
                                 }
                              } else {
                                 if (f.at<uchar>(k-1,l)!=INSIDE) {
                                    gradI.y=(float)((out.at<PixelT>(kp,lm)[color]-out.at<PixelT>(km-1,lm)[color]));
                                 } else {
                                    gradI.y=0;
                                 }
                              }
                              Ia[color] += (float)w * (float)(out.at<PixelT>(k-1,l-1)[color]);
                              Jx[color] -= (float)w * (float)(gradI.x*r.x);
                              Jy[color] -= (float)w * (float)(gradI.y*r.y);
                              s[color]  += w;
                           }
                        }
                     }
                  }
               }
               for (color=0; color<=2; color++) {
                  sat = (float)(Ia[color]/s[color]+(Jx[color]+Jy[color])/(sqrt(Jx[color]*Jx[color]+Jy[color]*Jy[color])+1.0e-20f));
                  out.at<PixelT>(i-1,j-1)[color] = round_cast<uchar>(sat);
               }

               f.at<uchar>(i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }

   } else if (out.channels()==1) {

      while (Heap->Pop(&ii,&jj)) {

         f.at<uchar>(ii,jj) = KNOWN;
         for(q=0; q<4; q++) {
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>t.rows-1)||(j>t.cols-1)) continue;

            if (f.at<uchar>(i,j)==INSIDE) {
               dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
               t.at<float>(i,j) = dist;

               for (color=0; color<=0; color++) {
                  cv::Point2f gradI,gradT,r;
                  float Ia=0,Jx=0,Jy=0,s=1.0e-20f,w,dst,lev,dir,sat;

                  if (f.at<uchar>(i,j+1)!=INSIDE) {
                     if (f.at<uchar>(i,j-1)!=INSIDE) {
                        gradT.x=(float)((t.at<float>(i,j+1)-t.at<float>(i,j-1)))*0.5f;
                     } else {
                        gradT.x=(float)((t.at<float>(i,j+1)-t.at<float>(i,j)));
                     }
                  } else {
                     if (f.at<uchar>(i,j-1)!=INSIDE) {
                        gradT.x=(float)((t.at<float>(i,j)-t.at<float>(i,j-1)));
                     } else {
                        gradT.x=0;
                     }
                  }
                  if (f.at<uchar>(i+1,j)!=INSIDE) {
                     if (f.at<uchar>(i-1,j)!=INSIDE) {
                        gradT.y=(float)((t.at<float>(i+1,j)-t.at<float>(i-1,j)))*0.5f;
                     } else {
                        gradT.y=(float)((t.at<float>(i+1,j)-t.at<float>(i,j)));
                     }
                  } else {
                     if (f.at<uchar>(i-1,j)!=INSIDE) {
                        gradT.y=(float)((t.at<float>(i,j)-t.at<float>(i-1,j)));
                     } else {
                        gradT.y=0;
                     }
                  }
                  for (k=i-range; k<=i+range; k++) {
                     int km=k-1+(k==1),kp=k-1-(k==t.rows-2);
                     for (l=j-range; l<=j+range; l++) {
                        int lm=l-1+(l==1),lp=l-1-(l==t.cols-2);
                        if (k>0&&l>0&&k<t.rows-1&&l<t.cols-1) {
                           if ((f.at<uchar>(k,l)!=INSIDE)&&
                               ((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                              r.y     = (float)(i-k);
                              r.x     = (float)(j-l);

                              dst = (float)(1./(VectorLength(r)*sqrt(VectorLength(r))));
                              lev = (float)(1./(1+fabs(t.at<float>(k,l)-t.at<float>(i,j))));

                              dir=VectorScalMult(r,gradT);
                              if (fabs(dir)<=0.01) dir=0.000001f;
                              w = (float)fabs(dst*lev*dir);

                              if (f.at<uchar>(k,l+1)!=INSIDE) {
                                 if (f.at<uchar>(k,l-1)!=INSIDE) {
                                    gradI.x=(float)((out.at<data_type>(km,lp+1)-out.at<data_type>(km,lm-1)))*2.0f;
                                 } else {
                                    gradI.x=(float)((out.at<data_type>(km,lp+1)-out.at<data_type>(km,lm)));
                                 }
                              } else {
                                 if (f.at<uchar>(k,l-1)!=INSIDE) {
                                    gradI.x=(float)((out.at<data_type>(km,lp)-out.at<data_type>(km,lm-1)));
                                 } else {
                                    gradI.x=0;
                                 }
                              }
                              if (f.at<uchar>(k+1,l)!=INSIDE) {
                                 if (f.at<uchar>(k-1,l)!=INSIDE) {
                                    gradI.y=(float)((out.at<data_type>(kp+1,lm)-out.at<data_type>(km-1,lm)))*2.0f;
                                 } else {
                                    gradI.y=(float)((out.at<data_type>(kp+1,lm)-out.at<data_type>(km,lm)));
                                 }
                              } else {
                                 if (f.at<uchar>(k-1,l)!=INSIDE) {
                                    gradI.y=(float)((out.at<data_type>(kp,lm)-out.at<data_type>(km-1,lm)));
                                 } else {
                                    gradI.y=0;
                                 }
                              }
                              Ia += (float)w * (float)(out.at<data_type>(k-1,l-1));
                              Jx -= (float)w * (float)(gradI.x*r.x);
                              Jy -= (float)w * (float)(gradI.y*r.y);
                              s  += w;
                           }
                        }
                     }
                  }
                  sat = (float)(Ia/s+(Jx+Jy)/(sqrt(Jx*Jx+Jy*Jy)+1.0e-20f));
                  out.at<data_type>(i-1,j-1) = round_cast<data_type>(sat);
               }

               f.at<uchar>(i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }
   }
#endif
}

template <typename data_type>
static void
icvNSInpaintFMM(Mat &f, Mat &t, Mat &out, int range, CvPriorityQueueFloat *Heap) {
#if CV_SIMD128
   icvNSInpaintFMM_simd<data_type>(f, t, out, range, Heap);
#else
   int i = 0, j = 0, ii = 0, jj = 0, k, l, q, color = 0;
   float dist;

   if (out.channels()==3) {
      typedef Vec<uchar, 3> PixelT;

      while (Heap->Pop(&ii,&jj)) {

         f.at<uchar>(ii,jj) = KNOWN;
         for(q=0; q<4; q++) {
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>t.rows-1)||(j>t.cols-1)) continue;

            if (f.at<uchar>(i,j)==INSIDE) {
               dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
               t.at<float>(i,j) = dist;

               cv::Point2f gradI,r;
               float Ia[3]={0,0,0};
               float s[3]={1.0e-20f,1.0e-20f,1.0e-20f};
               float w,dst,dir;

               for (k=i-range; k<=i+range; k++) {
                  int km=k-1+(k==1),kp=k-1-(k==f.rows-2);
                  for (l=j-range; l<=j+range; l++) {
                     int lm=l-1+(l==1),lp=l-1-(l==f.cols-2);
                     if (k>0&&l>0&&k<f.rows-1&&l<f.cols-1) {
                        if ((f.at<uchar>(k,l)!=INSIDE)&&
                            ((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                           for (color=0; color<=2; color++) {
                              r.y=(float)(k-i);
                              r.x=(float)(l-j);

                              dst = 1/(VectorLength(r)*VectorLength(r)+1);

                              if (f.at<uchar>(k+1,l)!=INSIDE) {
                                 if (f.at<uchar>(k-1,l)!=INSIDE) {
                                    gradI.x=(float)(abs(out.at<PixelT>(kp+1,lm)[color]-out.at<PixelT>(kp,lm)[color])+
                                                    abs(out.at<PixelT>(kp,lm)[color]-out.at<PixelT>(km-1,lm)[color]));
                                 } else {
                                    gradI.x=(float)(abs(out.at<PixelT>(kp+1,lm)[color]-out.at<PixelT>(kp,lm)[color]))*2.0f;
                                 }
                              } else {
                                 if (f.at<uchar>(k-1,l)!=INSIDE) {
                                    gradI.x=(float)(abs(out.at<PixelT>(kp,lm)[color]-out.at<PixelT>(km-1,lm)[color]))*2.0f;
                                 } else {
                                    gradI.x=0;
                                 }
                              }
                              if (f.at<uchar>(k,l+1)!=INSIDE) {
                                 if (f.at<uchar>(k,l-1)!=INSIDE) {
                                    gradI.y=(float)(abs(out.at<PixelT>(km,lp+1)[color]-out.at<PixelT>(km,lm)[color])+
                                                    abs(out.at<PixelT>(km,lm)[color]-out.at<PixelT>(km,lm-1)[color]));
                                 } else {
                                    gradI.y=(float)(abs(out.at<PixelT>(km,lp+1)[color]-out.at<PixelT>(km,lm)[color]))*2.0f;
                                 }
                              } else {
                                 if (f.at<uchar>(k,l-1)!=INSIDE) {
                                    gradI.y=(float)(abs(out.at<PixelT>(km,lm)[color]-out.at<PixelT>(km,lm-1)[color]))*2.0f;
                                 } else {
                                    gradI.y=0;
                                 }
                              }

                              gradI.x=-gradI.x;
                              dir=VectorScalMult(r,gradI);

                              if (fabs(dir)<=0.01) {
                                 dir=0.000001f;
                              } else {
                                 dir = (float)fabs(VectorScalMult(r,gradI)/sqrt(VectorLength(r)*VectorLength(gradI)));
                              }
                              w = dst*dir;
                              Ia[color] += (float)w * (float)(out.at<PixelT>(k-1,l-1)[color]);
                              s[color]  += w;
                           }
                        }
                     }
                  }
               }
               for (color=0; color<=2; color++) {
                  out.at<PixelT>(i-1,j-1)[color] = cv::saturate_cast<uchar>((double)Ia[color]/s[color]);
               }

               f.at<uchar>(i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }

   } else if (out.channels()==1) {

      while (Heap->Pop(&ii,&jj)) {

         f.at<uchar>(ii,jj) = KNOWN;
         for(q=0; q<4; q++) {
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>t.rows-1)||(j>t.cols-1)) continue;

            if (f.at<uchar>(i,j)==INSIDE) {
               dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
               t.at<float>(i,j) = dist;

               {
                  cv::Point2f gradI,r;
                  float Ia=0,s=1.0e-20f,w,dst,dir;

                  for (k=i-range; k<=i+range; k++) {
                     int km=k-1+(k==1),kp=k-1-(k==t.rows-2);
                     for (l=j-range; l<=j+range; l++) {
                        int lm=l-1+(l==1),lp=l-1-(l==t.cols-2);
                        if (k>0&&l>0&&k<t.rows-1&&l<t.cols-1) {
                           if ((f.at<uchar>(k,l)!=INSIDE)&&
                               ((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                              r.y=(float)(i-k);
                              r.x=(float)(j-l);

                              dst = 1/(VectorLength(r)*VectorLength(r)+1);

                              if (f.at<uchar>(k+1,l)!=INSIDE) {
                                 if (f.at<uchar>(k-1,l)!=INSIDE) {
                                    gradI.x=(float)(std::abs(out.at<data_type>(kp+1,lm)-out.at<data_type>(kp,lm))+
                                                    std::abs(out.at<data_type>(kp,lm)-out.at<data_type>(km-1,lm)));
                                 } else {
                                    gradI.x=(float)(std::abs(out.at<data_type>(kp+1,lm)-out.at<data_type>(kp,lm)))*2.0f;
                                 }
                              } else {
                                 if (f.at<uchar>(k-1,l)!=INSIDE) {
                                    gradI.x=(float)(std::abs(out.at<data_type>(kp,lm)-out.at<data_type>(km-1,lm)))*2.0f;
                                 } else {
                                    gradI.x=0;
                                 }
                              }
                              if (f.at<uchar>(k,l+1)!=INSIDE) {
                                 if (f.at<uchar>(k,l-1)!=INSIDE) {
                                    gradI.y=(float)(std::abs(out.at<data_type>(km,lp+1)-out.at<data_type>(km,lm))+
                                                    std::abs(out.at<data_type>(km,lm)-out.at<data_type>(km,lm-1)));
                                 } else {
                                    gradI.y=(float)(std::abs(out.at<data_type>(km,lp+1)-out.at<data_type>(km,lm)))*2.0f;
                                 }
                              } else {
                                 if (f.at<uchar>(k,l-1)!=INSIDE) {
                                    gradI.y=(float)(std::abs(out.at<data_type>(km,lm)-out.at<data_type>(km,lm-1)))*2.0f;
                                 } else {
                                    gradI.y=0;
                                 }
                              }

                              gradI.x=-gradI.x;
                              dir=VectorScalMult(r,gradI);

                              if (fabs(dir)<=0.01) {
                                 dir=0.000001f;
                              } else {
                                 dir = (float)fabs(VectorScalMult(r,gradI)/sqrt(VectorLength(r)*VectorLength(gradI)));
                              }
                              w = dst*dir;
                              Ia += (float)w * (float)(out.at<data_type>(k-1,l-1));
                              s  += w;
                           }
                        }
                     }
                  }
                  out.at<data_type>(i-1,j-1) = cv::saturate_cast<data_type>((double)Ia/s);
               }

               f.at<uchar>(i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }

   }
#endif
}

#define SET_BORDER1_C1(image,type,value) {\
      int i,j;\
      for(j=0; j<image.cols; j++) {\
         image.at<type>(0,j) = value;\
      }\
      for (i=1; i<image.rows-1; i++) {\
         image.at<type>(i,0) = image.at<type>(i,image.cols-1) = value;\
      }\
      for(j=0; j<image.cols; j++) {\
         image.at<type>(erows-1,j) = value;\
      }\
   }

static void
icvInpaint( const Mat &input_img, const Mat &inpaint_mask, Mat &output_img,
           double inpaintRange, int flags )
{
    cv::Mat mask, band, t, out;
    cv::Ptr<CvPriorityQueueFloat> Heap, Out;
    cv::Mat el_range, el_cross; // structuring elements for dilate

    int range=cvRound(inpaintRange);
    int erows, ecols;
    int maskRowMin = INT_MAX, maskRowMax = -1, maskColMin = INT_MAX, maskColMax = -1;

    if((input_img.size() != output_img.size()) || (input_img.size() != inpaint_mask.size()))
        CV_Error( cv::Error::StsUnmatchedSizes, "All the input and output images must have the same size" );

    if( (input_img.type() != CV_8U &&
         input_img.type() != CV_16U &&
         input_img.type() != CV_32F &&
        input_img.type() != CV_8UC3) ||
        (input_img.type() != output_img.type()) )
        CV_Error( cv::Error::StsUnsupportedFormat,
        "8-bit, 16-bit unsigned or 32-bit float 1-channel and 8-bit 3-channel input/output images are supported" );

    if( inpaint_mask.type() != CV_8UC1 )
        CV_Error( cv::Error::StsUnsupportedFormat, "The mask must be 8-bit 1-channel image" );

    range = MAX(range,1);
    range = MIN(range,100);

    ecols = input_img.cols + 2;
    erows = input_img.rows + 2;

    t.create(erows, ecols, CV_32FC1);
    band.create(erows, ecols, CV_8UC1);
    mask.create(erows, ecols, CV_8UC1);
    el_cross = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3), cv::Point(1, 1));

    input_img.copyTo( output_img );
    mask.setTo(Scalar(KNOWN,0,0,0));
    for (int i = 0; i < inpaint_mask.rows; i++) {
        const uchar* src = inpaint_mask.ptr<uchar>(i);
        uchar* dst = mask.ptr<uchar>(i + 1) + 1;
        for (int j = inpaintSkipBlocksBelow(src, 0, inpaint_mask.cols, 1); j < inpaint_mask.cols; j = inpaintSkipBlocksBelow(src, j, inpaint_mask.cols, 1)) {
            for (int blockEnd = std::min(j + VTraits<v_uint8>::vlanes(), inpaint_mask.cols); j < blockEnd; j++)
                if (src[j] != 0) {
                    dst[j] = INSIDE;
                    maskRowMin = std::min(maskRowMin, i); maskRowMax = i;
                    maskColMin = std::min(maskColMin, j); maskColMax = std::max(maskColMax, j);
                }
        }
    }
    SET_BORDER1_C1(mask,uchar,0);
    t.setTo(Scalar(1.0e6f,0,0,0));
    band.setTo(Scalar::all(0));
    Rect bandRoi, rangeRoi;
    if (maskColMax >= 0) {
        bandRoi = Rect(maskColMin, maskRowMin, maskColMax - maskColMin + 3, maskRowMax - maskRowMin + 3) & Rect(0, 0, ecols, erows);
        rangeRoi = Rect(maskColMin + 1 - range, maskRowMin + 1 - range, maskColMax - maskColMin + 1 + 2 * range, maskRowMax - maskRowMin + 1 + 2 * range) & Rect(0, 0, ecols, erows);
        cv::dilate(mask(bandRoi), band(bandRoi), el_cross, cv::Point(1, 1));
        subtract(band(bandRoi), mask(bandRoi), band(bandRoi));
    }
    Heap=cv::makePtr<CvPriorityQueueFloat>();
    SET_BORDER1_C1(band,uchar,0);
    if (!Heap->Add(band))
        return;

    t.setTo(Scalar(0,0,0,0),band);

    if( flags == cv::INPAINT_TELEA )
    {
        out.create(erows, ecols, CV_8UC1);
        el_range = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * range + 1, 2 * range + 1));
        out.setTo(Scalar::all(0));
        if (maskColMax >= 0) {
            cv::dilate(mask(rangeRoi), out(rangeRoi), el_range);
            subtract(out(rangeRoi), mask(rangeRoi), out(rangeRoi));
            subtract(out(rangeRoi), band(rangeRoi), out(rangeRoi));
        }
        Out=cv::makePtr<CvPriorityQueueFloat>();
        if (!Out->Add(band))
            return;
        SET_BORDER1_C1(out,uchar,0);
        icvCalcFMM(out,t,Out,true);
        switch(output_img.depth())
        {
            case CV_8U:
                icvTeleaInpaintFMM<uchar>(mask,t,output_img,range,Heap);
                break;
            case CV_16U:
                icvTeleaInpaintFMM<ushort>(mask,t,output_img,range,Heap);
                break;
            case CV_32F:
                icvTeleaInpaintFMM<float>(mask,t,output_img,range,Heap);
                break;
            default:
                CV_Error( cv::Error::StsBadArg, "Unsupportedformat of the input image" );
        }
    }
    else if (flags == cv::INPAINT_NS) {
        switch(output_img.depth())
        {
            case CV_8U:
                icvNSInpaintFMM<uchar>(mask,t,output_img,range,Heap);
                break;
            case CV_16U:
                icvNSInpaintFMM<ushort>(mask,t,output_img,range,Heap);
                break;
            case CV_32F:
                icvNSInpaintFMM<float>(mask,t,output_img,range,Heap);
                break;
            default:
                CV_Error( cv::Error::StsBadArg, "Unsupported format of the input image" );
        }
    } else {
        CV_Error( cv::Error::StsBadArg, "The flags argument must be one of INPAINT_TELEA or INPAINT_NS" );
    }
}

void cv::inpaint( InputArray _src, InputArray _mask, OutputArray _dst,
                  double inpaintRange, int flags )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat(), mask = _mask.getMat();
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();
    icvInpaint( src, mask, dst, inpaintRange, flags );
}
