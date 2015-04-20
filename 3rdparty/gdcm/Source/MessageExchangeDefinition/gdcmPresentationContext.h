/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPRESENTATIONCONTEXT_H
#define GDCMPRESENTATIONCONTEXT_H

#include "gdcmTypes.h"
#include "gdcmUIDs.h"

#include <vector>

namespace gdcm
{

/**
 * \brief PresentationContext
 * \see PresentationContextAC PresentationContextRQ
 */
class GDCM_EXPORT PresentationContext
{
public:
  PresentationContext();

  /// Initialize Presentation Context with AbstractSyntax set to asname
  /// and with a single TransferSyntax set to tsname (default to Implicit VR
  /// LittleEndian when not specified ).
  PresentationContext( UIDs::TSName asname,
    UIDs::TSName tsname = UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM );

  void SetAbstractSyntax( const char *as ) { AbstractSyntax = as; }
  const char *GetAbstractSyntax() const { return AbstractSyntax.c_str(); }

  void AddTransferSyntax( const char *tsstr );
  typedef std::vector<std::string> TransferSyntaxArrayType;
  typedef TransferSyntaxArrayType::size_type SizeType;
  const char *GetTransferSyntax(SizeType i) const { return TransferSyntaxes[i].c_str(); }
  SizeType GetNumberOfTransferSyntaxes() const { return TransferSyntaxes.size(); }

  void SetPresentationContextID( uint8_t id );
  uint8_t GetPresentationContextID() const;

  void Print(std::ostream &os) const;

  bool operator==(const PresentationContext & pc) const
    {
    assert( TransferSyntaxes.size() == 1 ); // TODO
    assert( pc.TransferSyntaxes.size() == 1 );
    return AbstractSyntax == pc.AbstractSyntax && TransferSyntaxes == pc.TransferSyntaxes;
    }

private:
  std::string AbstractSyntax;
  std::vector<std::string> TransferSyntaxes;
  uint8_t /*PresentationContext*/ID;
};

} // end namespace gdcm

#endif //GDCMPRESENTATIONCONTEXT_H
