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

#include <queue>
#include <type_traits>

#include "precomp.hpp"

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

typedef struct CvHeapElem
{
    float T;
    int i,j;
    int order;  // to keep insertion order

    bool operator > (const CvHeapElem& rhs) const {
        if (T > rhs.T) {
            return true;
        } else if (T < rhs.T) {
            return false;
        }
        return order > rhs.order;
    }
}
CvHeapElem;


class CvPriorityQueueFloat
{
private:
    CvPriorityQueueFloat(const CvPriorityQueueFloat & ); // copy disabled
    CvPriorityQueueFloat& operator=(const CvPriorityQueueFloat &); // assign disabled

protected:
    std::priority_queue<CvHeapElem, std::vector<CvHeapElem>,std::greater<CvHeapElem> > queue;
    int next_order;

public:
    bool Add(const Mat &f) {
        int i,j;
        for (i=0; i<f.rows; i++) {
            for (j=0; j<f.cols; j++) {
                if (f.at<uchar>(i, j)!=0) {
                    if (!Push(i,j,0)) return false;
                }
            }
        }
        return true;
    }

    bool Push(int i, int j, float T) {
        queue.push({T, i, j, next_order});
        ++next_order;
        return true;
    }

    bool Pop(int *i, int *j) {
        if (queue.empty()) {
            return false;
        }
        *i = queue.top().i;
        *j = queue.top().j;
        queue.pop();
        return true;
    }

    bool Pop(int *i, int *j, float *T) {
        if (queue.empty()) {
            return false;
        }
        *i = queue.top().i;
        *j = queue.top().j;
        *T = queue.top().T;
        queue.pop();
        return true;
    }

    CvPriorityQueueFloat(void) : queue(), next_order() {
    }
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
            dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                        FastMarching_solve(i+1,j,i,j-1,f,t),
                        FastMarching_solve(i-1,j,i,j+1,f,t),
                        FastMarching_solve(i+1,j,i,j+1,f,t));
            t.at<float>(i,j) = dist;
            f.at<uchar>(i,j) = BAND;
            Heap->Push(i,j,dist);
         }
      }
   }

   if (negate) {
      for (i=0; i<f.rows; i++) {
         for(j=0; j<f.cols; j++) {
            if (f.at<uchar>(i,j) == CHANGE) {
               f.at<uchar>(i,j) = KNOWN;
               t.at<float>(i,j) = -t.at<float>(i,j);
            }
         }
      }
   }
}

template <typename data_type>
static void
icvTeleaInpaintFMM(Mat &f, Mat &t, Mat &out, int range, CvPriorityQueueFloat *Heap ) {
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
                  {
                  out.at<data_type>(i-1,j-1) = round_cast<data_type>(sat);
                  }
               }

               f.at<uchar>(i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }
   }
}

template <typename data_type>
static void
icvNSInpaintFMM(Mat &f, Mat &t, Mat &out, int range, CvPriorityQueueFloat *Heap) {
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

#define COPY_MASK_BORDER1_C1(src,dst,type) {\
      int i,j;\
      for (i=0; i<src.rows; i++) {\
         for(j=0; j<src.cols; j++) {\
            if (src.at<type>(i,j)!=0)\
               dst.at<type>(i+1,j+1) = INSIDE;\
         }\
      }\
   }

static void
icvInpaint( const Mat &input_img, const Mat &inpaint_mask, Mat &output_img,
           double inpaintRange, int flags )
{
    cv::Mat mask, band, f, t, out;
    cv::Ptr<CvPriorityQueueFloat> Heap, Out;
    cv::Mat el_range, el_cross; // structuring elements for dilate

    int range=cvRound(inpaintRange);
    int erows, ecols;

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

    f.create(erows, ecols, CV_8UC1);
    t.create(erows, ecols, CV_32FC1);
    band.create(erows, ecols, CV_8UC1);
    mask.create(erows, ecols, CV_8UC1);
    el_cross = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3), cv::Point(1, 1));

    input_img.copyTo( output_img );
    mask.setTo(Scalar(KNOWN,0,0,0));
    COPY_MASK_BORDER1_C1(inpaint_mask,mask,uchar);
    SET_BORDER1_C1(mask,uchar,0);
    f.setTo(Scalar(KNOWN,0,0,0));
    t.setTo(Scalar(1.0e6f,0,0,0));
    cv::dilate(mask, band, el_cross, cv::Point(1, 1));
    Heap=cv::makePtr<CvPriorityQueueFloat>();
    subtract(band, mask, band);
    SET_BORDER1_C1(band,uchar,0);
    if (!Heap->Add(band))
        return;

    f.setTo(Scalar(BAND,0,0,0),band);
    f.setTo(Scalar(INSIDE,0,0,0),mask);
    t.setTo(Scalar(0,0,0,0),band);

    if( flags == cv::INPAINT_TELEA )
    {
        out.create(erows, ecols, CV_8UC1);
        el_range = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * range + 1, 2 * range + 1));
        cv::dilate(mask, out, el_range);
        subtract(out, mask, out);
        Out=cv::makePtr<CvPriorityQueueFloat>();
        if (!Out->Add(band))
            return;
        subtract(out, band, out);
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
