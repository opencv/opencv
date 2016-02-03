/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPRINTER_H
#define GDCMPRINTER_H

// TODO Class to implement printing
// Since DICOM does printing ?
// Also I would like to encapsulate the IsCharacterPrintable thing
// (to avoid printing \0 and other weird characters)
// \todo I still need to implement skiping of group (shadow)
// need to implement longer field to read

/*
 * Output:
 * For ASCII:
 * Typically will look like:
 * [ORIGINAL\PRIMARY\OTHER]
 * If a non printable character is found: RED and INVERSE is used:
 * [                .]
 *
 * when the VR is not found (file or dict), we check if we can print the output:
 * on success ASCII mode is used, on failure the output is printed a series of bytes
 *
 * Special case when the data element is empty:
 * INVERSE << (no value)
 *
 * retired public element are printed in red and underline
 * unknown private element are printed in RED followed by 'UNKNOWN'
 *
 * Correct VR is printed in green just after the found VR
 *
 * length of data element is printed in bytes, followed by the VM, a green VM is appended
 * if this is not compatible
 */
#include "gdcmFile.h"
#include "gdcmDataElement.h"

namespace gdcm
{

class DataSet;
class DictEntry;
class Dicts;
/**
 * \brief Printer class
 */
// It's a sink there is no output
class GDCM_EXPORT Printer
{
public:
  Printer();
  ~Printer();

  /// Set file
  void SetFile(File const &f) { F = &f; }

  /// Set color mode or not
  void SetColor(bool c);

  typedef enum {
    VERBOSE_STYLE = 0, // GDCM Legacy VERBOSE one
    CONDENSED_STYLE, //
    // Ok I am missing voc here ...better naming would be nice
    XML // sure why not
  } PrintStyles;

  /// Set PrintStyle value
  void SetStyle(PrintStyles ps) {
    PrintStyle = ps;
  }
  /// Get PrintStyle value
  PrintStyles GetPrintStyle() const {
    return PrintStyle;
    }

  /// Print
  void Print(std::ostream& os);

  /// Print an individual dataset
  void PrintDataSet(const DataSet &ds, std::ostream& os, const std::string &s = "");

protected:
  VR PrintDataElement(std::ostringstream & os, const Dicts &dicts, const DataSet & ds, const DataElement &de, std::ostream &out, std::string const & indent );
void PrintSQ(const SequenceOfItems *sqi, std::ostream & os, std::string const & indent);

  PrintStyles PrintStyle;
  const File *F;
  VL MaxPrintLength;
};

} // end namespace gdcm

#endif //GDCMPRINTER_H
