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

#include "precomp.hpp"
#include "opencv2/video/onlineMIL.hpp"

namespace cv
{

//implementations for strong classifier

ClfMilBoost::Params::Params()
{
  _numSel = 250;
  _numFeat = 250;
}

ClfMilBoost::ClfMilBoost()
{
  _myParams = ClfMilBoost::Params();
  _numsamples = 0;
}

void ClfMilBoost::init( const ClfMilBoost::Params &parameters )
{
  _myParams = parameters;
}

void ClfMilBoost::update( const Mat& posx, const Mat& negx )
{
}

std::vector<float> ClfMilBoost::classify( const Mat& x, bool logR )
{
  std::vector<float> prob;
  return prob;
}

//implementations for weak classifier

ClfOnlineStump::ClfOnlineStump()
{
  _trained = false;
  _ind = -1;
  init();
}

ClfOnlineStump::ClfOnlineStump( int ind )
{
  _trained = false;
  _ind = ind;
  init();
}
void ClfOnlineStump::init()
{
  _mu0 = 0;
  _mu1 = 0;
  _sig0 = 1;
  _sig1 = 1;
  _lRate = 0.85f;
  _trained = false;
}

void ClfOnlineStump::update( const Mat& posx, const Mat& negx, const Mat_<float> & posw, const Mat_<float> & negw )
{
  float posmu = 0.0, negmu = 0.0;
  if( posx.cols > 0 )
    posmu = mean( posx.col( _ind ) )[0];
  if( negx.cols > 0 )
    negmu = mean( negx.col( _ind ) )[0];

  if( _trained )
  {
    if( posx.cols > 0 )
    {
      _mu1 = ( _lRate * _mu1 + ( 1 - _lRate ) * posmu );
      cv::Mat diff = posx.col( _ind ) - _mu1;
      _sig1 = _lRate * _sig1 + ( 1 - _lRate ) * cv::mean( diff.mul( diff ) )[0];
    }
    if( negx.cols > 0 )
    {
      _mu0 = ( _lRate * _mu0 + ( 1 - _lRate ) * negmu );
      cv::Mat diff = negx.col( _ind ) - _mu0;
      _sig0 = _lRate * _sig0 + ( 1 - _lRate ) * cv::mean( diff.mul( diff ) )[0];
    }

    _q = ( _mu1 - _mu0 ) / 2;
    _s = sign( _mu1 - _mu0 );
    _log_n0 = std::log( float( 1.0f / pow( _sig0, 0.5f ) ) );
    _log_n1 = std::log( float( 1.0f / pow( _sig1, 0.5f ) ) );
    //_e1 = -1.0f/(2.0f*_sig1+1e-99f);
    //_e0 = -1.0f/(2.0f*_sig0+1e-99f);
    _e1 = -1.0f / ( 2.0f * _sig1 + std::numeric_limits<float>::min() );
    _e0 = -1.0f / ( 2.0f * _sig0 + std::numeric_limits<float>::min() );

  }
  else
  {
    _trained = true;
    if( posx.cols > 0 )
    {
      _mu1 = posmu;
      cv::Scalar scal_mean, scal_std_dev;
      cv::meanStdDev( posx.col( _ind ), scal_mean, scal_std_dev );
      _sig1 = scal_std_dev[0] * scal_std_dev[0] + 1e-9f;
    }

    if( negx.cols > 0 )
    {
      _mu0 = negmu;
      cv::Scalar scal_mean, scal_std_dev;
      cv::meanStdDev( negx.col( _ind ), scal_mean, scal_std_dev );
      _sig0 = scal_std_dev[0] * scal_std_dev[0] + 1e-9f;
    }

    _q = ( _mu1 - _mu0 ) / 2;
    _s = sign( _mu1 - _mu0 );
    _log_n0 = std::log( float( 1.0f / pow( _sig0, 0.5f ) ) );
    _log_n1 = std::log( float( 1.0f / pow( _sig1, 0.5f ) ) );
    //_e1 = -1.0f/(2.0f*_sig1+1e-99f);
    //_e0 = -1.0f/(2.0f*_sig0+1e-99f);
    _e1 = -1.0f / ( 2.0f * _sig1 + std::numeric_limits<float>::min() );
    _e0 = -1.0f / ( 2.0f * _sig0 + std::numeric_limits<float>::min() );
  }
}

bool ClfOnlineStump::classify( const Mat& x, int i )
{
  float xx = x.at<float>( i, _ind );
  double log_p0 = ( xx - _mu0 ) * ( xx - _mu0 ) * _e0 + _log_n0;
  double log_p1 = ( xx - _mu1 ) * ( xx - _mu1 ) * _e1 + _log_n1;
  return log_p1 > log_p0;
}

float ClfOnlineStump::classifyF( const Mat& x, int i )
{
  float xx = x.at<float>( i, _ind );
  double log_p0 = ( xx - _mu0 ) * ( xx - _mu0 ) * _e0 + _log_n0;
  double log_p1 = ( xx - _mu1 ) * ( xx - _mu1 ) * _e1 + _log_n1;
  return float( log_p1 - log_p0 );
}

} /* namespace cv */
