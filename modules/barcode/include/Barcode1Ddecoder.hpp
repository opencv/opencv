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

#ifndef BARCODE1DDECODER_H_
#define BARCODE1DDECODER_H_

#ifdef __cplusplus

#include <stdio.h>
#include <iostream>

#include "opencv2/core.hpp"

#include "Symbologies.hpp"


namespace cv {


class CV_EXPORTS_W Barcode1Ddecoder : public virtual Algorithm
{
 public:
  Mat barcode;//ROI barcode
  int symbology_type;//i.e. UPC_A, CODE_128, etc.
  int lenght;//optional: number of digit in barcode given in input
  std::vector<int> decoded_digit;//indexes of decoded digit
  Symbology* symbology;//we can set a specific symbology or try all.

  //create the specific barcode, we can set symbology and lenght of barcode
  CV_WRAP static Ptr<Barcode1Ddecoder> create( const std::string& decoder_type, const Mat& barcode, 
                                const int symbology_type = -1, const int lenght = -1);
  /* This method returns the string of decoded barcode, here is implemented the decoding algorithm
   * it matches the pixels of barcode with the symbols of symbology and takes a vector that represents 
   * the indexes of found symbols. 
   * this vector is then decoded by getDecoding of Symbology class.
   */
  virtual std::string decodeBarcode() = 0;
  virtual ~Barcode1Ddecoder();
};
}

#endif /* __cplusplus */

#endif /* BARCODE1DDECODER_H */
