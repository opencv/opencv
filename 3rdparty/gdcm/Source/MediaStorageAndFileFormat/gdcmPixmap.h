/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPIXMAP_H
#define GDCMPIXMAP_H

#include "gdcmBitmap.h"
#include "gdcmCurve.h"
#include "gdcmIconImage.h"
#include "gdcmOverlay.h"

namespace gdcm
{

/**
 * \brief Pixmap class
 * A bitmap based image. Used as parent for both IconImage and the main Pixel Data Image
 * It does not contains any World Space information (IPP, IOP)
 *
 * \see PixmapReader
 */
class GDCM_EXPORT Pixmap : public Bitmap
{
public:
  Pixmap();
  ~Pixmap();
  void Print(std::ostream &) const;

  /// returns if Overlays are stored in the unused bit of the pixel data:
  bool AreOverlaysInPixelData() const;

  /// Curve: group 50xx
  Curve& GetCurve(size_t i = 0) {
    assert( i < Curves.size() );
    return Curves[i];
  }
  const Curve& GetCurve(size_t i = 0) const {
    assert( i < Curves.size() );
    return Curves[i];
  }
  size_t GetNumberOfCurves() const { return Curves.size(); }
  void SetNumberOfCurves(size_t n) { Curves.resize(n); }

  /// Overlay: group 60xx
  Overlay& GetOverlay(size_t i = 0) {
    assert( i < Overlays.size() );
    return Overlays[i];
  }
  const Overlay& GetOverlay(size_t i = 0) const {
    assert( i < Overlays.size() );
    return Overlays[i];
  }
  size_t GetNumberOfOverlays() const { return Overlays.size(); }
  void SetNumberOfOverlays(size_t n) { Overlays.resize(n); }
  void RemoveOverlay(size_t i) {
    assert( i < Overlays.size() );
    Overlays.erase( Overlays.begin() + i );
  }

  /// Set/Get Icon Image
  const IconImage &GetIconImage() const { return *Icon; }
  IconImage &GetIconImage() { return *Icon; }
  void SetIconImage(IconImage const &ii) { Icon = ii; }

//private:
protected:
  std::vector<Overlay>  Overlays;
  std::vector<Curve>  Curves;
  SmartPointer<IconImage> Icon;
};

} // end namespace gdcm

#endif //GDCMPIXMAP_H
