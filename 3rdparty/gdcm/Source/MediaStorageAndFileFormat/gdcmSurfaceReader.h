/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSURFACEREADER_H
#define GDCMSURFACEREADER_H

#include <gdcmSegmentReader.h>
#include <gdcmSurface.h>

namespace gdcm
{

/**
  * \brief  This class defines a SURFACE IE reader.
  * It reads surface mesh module attributes.
  *
  * \see  PS 3.3 A.1.2.18 , A.57 and C.27
  */
class GDCM_EXPORT SurfaceReader : public SegmentReader
{
public:
    SurfaceReader();

    virtual ~SurfaceReader();

    /// Read
    virtual bool Read();

    unsigned long GetNumberOfSurfaces() const;

  protected:

    bool ReadSurfaces();

    bool ReadSurface(const Item & surfaceItem, const unsigned long idx);

    bool ReadPointMacro(SmartPointer< Surface > surface, const DataSet & surfaceDS);
};

}

#endif // GDCMSURFACEREADER_H
