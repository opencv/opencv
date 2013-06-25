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

#include "precomp.hpp"
#include "symbologies.hpp"

namespace cv
{

namespace barcode {

int Symbology::symbology = -1;

Ptr<Symbology> Symbology::create(const int sym)
{
  CV_Assert(symbology == CODE_128
         ||
         symbology == UPC_A);

  symbology = sym;
  switch (symbology) {
    case UPC_A:
      return  (Ptr<SymbologyUPCA>) new SymbologyUPCA();
      break;
    default:
      break;
  }
  return NULL;
}

bool Symbology::verifyCheckCode(const std::vector<int>& index_of_digit) const
{
  return (index_of_digit.at(CHECKCODE_POSITION_BARCODE) == getCheckCodeIndex(index_of_digit));
}


Symbology::~Symbology() {
  delete(this);
}

const int SymbologyUPCA::symbology = UPC_A;
const int SymbologyUPCA::barcode_length = 15;
const bool SymbologyUPCA::fixed_length = true;


//this is an example of a Symbology
SymbologyUPCA::SymbologyUPCA() {

  encoding.resize(ALPHABET_LENGTH);
  encoding.at(0) = "0";
  encoding.at(1) = "1";
  encoding.at(2) = "2";
  encoding.at(3) = "3";
  encoding.at(4) = "4";
  encoding.at(5) = "5";
  encoding.at(6) = "6";
  encoding.at(7) = "7";
  encoding.at(8) = "8";
  encoding.at(9) = "9";
  encoding.at(10) = ""; //BORDER BAR
  encoding.at(11) = ""; //CENTRAL BAR

  bars_pattern_left.resize(ALPHABET_LENGTH);
  //left side
  bars_pattern_left.at(0) = "0001101"; //00
  bars_pattern_left.at(1) = "0011001"; //01
  bars_pattern_left.at(2) = "0010011"; //02
  bars_pattern_left.at(3) = "0111101"; //03
  bars_pattern_left.at(4) = "0100011"; //04
  bars_pattern_left.at(5) = "0110001"; //05
  bars_pattern_left.at(6) = "0101111"; //06
  bars_pattern_left.at(7) = "0111011"; //07
  bars_pattern_left.at(8) = "0110111"; //08
  bars_pattern_left.at(9) = "0001011"; //09
  bars_pattern_left.at(10) = "101"; //border bar
  bars_pattern_left.at(11) = "01010"; //central bar

  bars_pattern_right.resize(ALPHABET_LENGTH);
  //right side
  bars_pattern_right.at(0) = "1110010"; //00
  bars_pattern_right.at(1) = "1100110"; //01
  bars_pattern_right.at(2) = "1101100"; //02
  bars_pattern_right.at(3) = "1000010"; //03
  bars_pattern_right.at(4) = "1011100"; //04
  bars_pattern_right.at(5) = "1001110"; //05
  bars_pattern_right.at(6) = "1010000"; //06
  bars_pattern_right.at(7) = "1000100"; //07
  bars_pattern_right.at(8) = "1001000"; //08
  bars_pattern_right.at(9) = "1110100"; //09
  bars_pattern_right.at(10) = "101"; //border bar
  bars_pattern_right.at(11) = "01010"; //central bar
  
  //for each position of a digit in the barcode there are only some possibile candidate symbols that can match
  std::vector<std::string>::iterator it_symbol;
  for (int i = 0; i < barcode_length; ++i)
  {
    Digit digit;
    digit.position = i;
    if (i == STARTCODE_POSITION_BARCODE || i == ENDCODE_POSITION_BARCODE)
    {
	  int indexes[] = {BORDER_BAR_INDEX_ALPHABET};
	  std::vector<int> indexes_vec (indexes, indexes + sizeof(indexes) / sizeof(int) );
	  digit.candidate_symbols = indexes_vec;
    }
    else if (i == CENTER_BAR_INDEX_ALPHABET)
    {
	  int indexes[] = {CENTER_BAR_INDEX_ALPHABET};
	  std::vector<int> indexes_vec (indexes, indexes + sizeof(indexes) / sizeof(int) );
	  digit.candidate_symbols = indexes_vec;
    }
    else
    {
	  int indexes[] = {0,1,2,3,4,5,6,7,8,9};
	  std::vector<int> indexes_vec (indexes, indexes + sizeof(indexes) / sizeof(int) );
	  digit.candidate_symbols = indexes_vec;
    }
    digits.push_back(digit);
  }
}


SymbologyUPCA::~SymbologyUPCA() {
  delete(this);
}

std::string SymbologyUPCA::getDecoding(const std::vector<int>& index_of_digit)
{
  (void) index_of_digit;
  std::string decoding;
  /*
   * logic for decoding
   */
  return decoding;
}

int SymbologyUPCA::getCheckCodeIndex(const std::vector<int>& index_of_digit) const
{
  (void) index_of_digit;
  int checkcode = 0;
  /*
   * logic to compute checkcode
   */
  return checkcode;
}

std::string SymbologyUPCA::getCheckCodeValue(const std::vector<int>& index_of_digit)
{
  unsigned int digit = getCheckCodeIndex(index_of_digit);
  CV_Assert(digit < encoding.size());
  return encoding.at(digit);
}

int SymbologyUPCA::getBarcodeLength() const
{
  return barcode_length;	
}

}
}
