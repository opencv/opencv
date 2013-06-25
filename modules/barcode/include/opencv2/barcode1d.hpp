/*#**********************************************************************************************
** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
**
** By downloading, copying, installing or using the software you agree to this license.
** If you do not agree to this license, do not download, install,
** copy or use the software.
**
**
**                          License Agreement
**               For Open Source Computer Vision Library
**
** Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
** Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
** Third party copyrights are property of their respective owners.
** 
** Redistribution and use in source and binary forms, with or without modification,
** are permitted provided that the following conditions are met:
** 
**   * Redistributions of source code must retain the above copyright notice,
**     this list of conditions and the following disclaimer.
** 
**   * Redistributions in binary form must reproduce the above copyright notice,
**     this list of conditions and the following disclaimer in the documentation
**     and/or other materials provided with the distribution.
** 
**   * The name of the copyright holders may not be used to endorse or promote products
**     derived from this software without specific prior written permission.
** 
** This software is provided by the copyright holders and contributors "as is" and
** any express or implied warranties, including, but not limited to, the implied
** warranties of merchantability and fitness for a particular purpose are disclaimed.
** In no event shall the Intel Corporation or contributors be liable for any direct,
** indirect, incidental, special, exemplary, or consequential damages
** (including, but not limited to, procurement of substitute goods or services;
** loss of use, data, or profits; or business interruption) however caused
** and on any theory of liability, whether in contract, strict liability,
** or tort (including negligence or otherwise) arising in any way out of
** the use of this software, even if advised of the possibility of such damage.
**  
************************************************************************************************/
/*#**********************************************************************************************
**                Creation - enhancement process 2012-2013                                      *
**                                                                                              *
** Authors: Claudia Rapuano (c.rapuano@gmail.com), University La Sapienza di Roma, Rome, Italy  *
** 	        Stefano Fabri (s.fabri@email.it), Rome, Italy                                       *
************************************************************************************************/
#ifndef __OPENCV_BARCODE_1D_HPP__
#define __OPENCV_BARCODE_1D_HPP__

#include <stdio.h>
#include <iostream>

#include "opencv2/core.hpp"

namespace cv {

namespace barcode {

class CV_EXPORTS_W Detector1d : public virtual Algorithm{
 public:
  //create specific 1D barcode locator
  CV_WRAP static Ptr<Detector1d> create( const std::string& locator_type );
  //locate barcodes in image, return related rotated rect
  virtual void locate(const Mat& image, std::vector<RotatedRect>& barcodes) = 0;
  virtual ~Detector1d();
};

class CV_EXPORTS_W Decoder1d : public virtual Algorithm
{
 public:
  //create the specific barcode decoder
  CV_WRAP static Ptr<Decoder1d> create( const std::string& decoder_type);

  /* This method returns the string of decoded barcode, here is implemented the decoding algorithm
   * it matches the pixels of barcode with the symbols of symbology and takes a vector that represents 
   * the indexes of found symbols. 
   * this vector is then decoded by getDecoding of Symbology class.
   */
  virtual std::string decodeBarcode() = 0;
  virtual ~Decoder1d();
 protected:
  Mat barcode;//ROI barcode
  int length;//we can set number of digit in barcode
  std::vector<int> decoded_digit;//indexes of decoded digit
};

}
}

#endif /* __OPENCV_BARCODE_1D_HPP__ */

