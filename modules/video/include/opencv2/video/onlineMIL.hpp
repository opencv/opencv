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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_ONLINEMIL_HPP__
#define __OPENCV_ONLINEMIL_HPP__

#include "opencv2/core.hpp"
#include <limits>

namespace cv
{
//TODO based on the original implementation
//http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml

#define  sign(s)  ((s > 0 ) ? 1 : ((s<0) ? -1 : 0))

class ClfOnlineStump;

class ClfMilBoost
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    int _numSel;
    int _numFeat;
    float _lRate;
  };

  ClfMilBoost();
  ~ClfMilBoost();
  void init( const ClfMilBoost::Params &parameters = ClfMilBoost::Params() );
  void update( const Mat& posx, const Mat& negx );
  std::vector<float> classify( const Mat& x, bool logR = true );

  inline float sigmoid( float x )
  {
    return 1.0f / ( 1.0f + exp( -x ) );
  }

 private:
  uint _numsamples;
  ClfMilBoost::Params _myParams;
  std::vector<int> _selectors;
  std::vector<ClfOnlineStump*> _weakclf;
  uint _counter;

};

class ClfOnlineStump
{
 public:
  float _mu0, _mu1, _sig0, _sig1;
  float _q;
  int _s;
  float _log_n1, _log_n0;
  float _e1, _e0;
  float _lRate;

  ClfOnlineStump();
  ClfOnlineStump( int ind );
  void init();
  void update( const Mat& posx, const Mat& negx, const cv::Mat_<float> & posw = cv::Mat_<float>(), const cv::Mat_<float> & negw = cv::Mat_<float>() );
  bool classify( const Mat& x, int i );
  float classifyF( const Mat& x, int i );
  std::vector<float> classifySetF( const Mat& x );

 private:
  bool _trained;
  int _ind;

};

} /* namespace cv */

#endif
