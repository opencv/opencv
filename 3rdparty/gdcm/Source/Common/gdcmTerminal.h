/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTERMINAL_H
#define GDCMTERMINAL_H

#include "gdcmTypes.h"


namespace gdcm
{
/**
 * \brief Class for Terminal
 * Allow one to print in color in a shell
 * - support VT100 compatible shell
 * - win32 console
 */
//-----------------------------------------------------------------------------

namespace terminal
{
  typedef enum
    {
    CONSOLE = 0,
    VT100
    } Mode;
  typedef enum
    {
    black = 0,
    red,
    green,
    yellow, // brown ??
    blue,
    magenta,
    cyan,
    white
    } Color;
  typedef enum
    {
    reset     = 0,
    bright    = 1, // bold
    dim       = 2,
    underline = 3,
    blink     = 5,
    reverse   = 7,
    hidden    = 8
    } Attribute;
  GDCM_EXPORT std::string setattribute( Attribute att );
  GDCM_EXPORT std::string setfgcolor( Color c );
  GDCM_EXPORT std::string setbgcolor( Color c );
  GDCM_EXPORT void setmode( Mode m);
}

} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMTERMINAL_H
