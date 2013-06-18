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

#ifndef SYMBOLOGIES_HPP_
#define SYMBOLOGIES_HPP_

#ifdef __cplusplus

#include <stdio.h>
#include <vector>
#include <iostream>
#include "opencv2/core/types_c.h"

namespace cv {

/*
 * Here there are defined each Barcode symbology. (UPCA, CODE128, etc)
 * Each symbology extend the class Symbology where are defined common members
 * SymbologyUPCA is an example of barcode syntax
 */

class Symbology {
 public:
  //list of sybologies covered
  enum symbology_type{
   UPC_A,
   /*.
     .
     .
     */
   CODE_128,
   FIRST_TYPE = UPC_A,
   LAST_TYPE = CODE_128
  };

  //(*)for each position of a digit in the barcode there are only some possibile candidate symbols that can match
  /*****TODO We want to use a map?*****/
  struct Digit{
    int position;
    std::vector<int> candidate_symbols;
  };
  typedef std::vector<Digit> Digits;

  virtual ~Symbology();

  /* Decodes a vector of indexes of digits found by the decoder 
   * and matches each index with its value.
   * Here there is implemented matching logic.
   */
  virtual std::string getDecoding(const std::vector<int>& index_of_digit) = 0;
  //return index of checkcode
  virtual int getCheckCodeIndex(const std::vector<int>& index_of_digit) const = 0;
  //return value of checkcode
  virtual std::string getCheckCodeValue(const std::vector<int>& index_of_digit) = 0;
  //verify if checkdigit decoded is right
  bool verifyCheckCode(const std::vector<int>& index_of_digit) const;
  //create Symbology
  static Ptr<Symbology> create(const int sym);
  //set the number of digit in the barcode to decode
  void setBarcodeLenght(const int lenght){barcode_lenght = lenght;};
  int getBarcodeLenght(){return barcode_lenght;};
  //return the std::vector of digit, see (*)
  Digits getDigits(){return digits;};
//  Digit getDigit(const int i, const vector<int>& previous);
  
 protected:
  static int symbology;//type of symbology
  bool fixed_lenght;//true if the symbology has a fixed number of digit in the barcode
  static int barcode_lenght;
  int ALPHABET_LENGHT;//number of symbols the syntax alphabet
  static const int STARTCODE_POSITION_BARCODE = 1;
  int CHECKCODE_POSITION_BARCODE;
  int ENDCODE_POSITION_BARCODE;
  Digits digits;//vector of digits, see(*)

  Symbology(){};
};

//==========================================================================//

//This is an example of a Symbology
class SymbologyUPCA : public virtual Symbology {
 public:
  virtual ~SymbologyUPCA();

  static const int ENDCODE_POSITION_BARCODE = 14;
  static const int CHECKCODE_POSITION_BARCODE = 13;
  static const int CENTER_BAR_POSITION_BARCODE = 7;

  static const int CENTER_BAR_LENGHT = 5;
  static const int BORDER_BAR_LENGHT = 3;
  static const int DIGIT_LENGHT = 7;

  static const int BORDER_BAR_INDEX_ALPHABET = 10;
  static const int CENTER_BAR_INDEX_ALPHABET = 11;

  static const int ALPHABET_LENGHT = 12;

  static const int barcode_lenght = 15;
  static const bool fixed_lenght = true;


  /*****TODO We want to use a map?*****/
  std::vector<std::string> bars_pattern_left; //vector of bars (i.e. of a bars "1101001") 
  std::vector<std::string> bars_pattern_right; 
  /*****TODO I'm not sure if it's the better representation*****/
  /*****Maybe we can use hex representation like zxing with bit shift?*****/

  //vectors of encoding
  std::vector<std::string> encoding;

  SymbologyUPCA();
  std::string getDecoding(const std::vector<int>& index_of_digit);
  std::string getDecoding(int index_of_digit);
  int getCheckCodeIndex(const std::vector<int>& index_of_digit) const;
  std::string getCheckCodeValue(const std::vector<int>& index_of_digit);

 private:
};
}

#endif /* __cplusplus */

#endif /* SYMBOLOGIES_HPP_ */

//==========================================================================//
