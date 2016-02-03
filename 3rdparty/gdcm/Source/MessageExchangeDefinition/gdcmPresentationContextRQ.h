/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPRESENTATIONCONTEXTRQ_H
#define GDCMPRESENTATIONCONTEXTRQ_H

#include "gdcmTypes.h"
#include "gdcmAbstractSyntax.h"
#include "gdcmTransferSyntaxSub.h"
#include "gdcmDataSet.h"

namespace gdcm
{
class PresentationContext;
namespace network
{

/**
 * \brief PresentationContextRQ
 * Table 9-13
 * PRESENTATION CONTEXT ITEM FIELDS
 * \see PresentationContextAC
 */
class GDCM_EXPORT PresentationContextRQ
{
public:
  PresentationContextRQ();

  /// Initialize Presentation Context with AbstractSyntax set to asname
  /// and with a single TransferSyntax set to tsname (dfault to Implicit VR
  /// LittleEndian when not specified ).
  PresentationContextRQ( UIDs::TSName asname, UIDs::TSName tsname =
    UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM  );

  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;
  size_t Size() const;

  void SetAbstractSyntax( AbstractSyntax const & as );
  AbstractSyntax const &GetAbstractSyntax() const { return SubItems; }
  AbstractSyntax &GetAbstractSyntax() { return SubItems; }

  void AddTransferSyntax( TransferSyntaxSub const &ts );
  typedef std::vector<TransferSyntaxSub>::size_type SizeType;
  TransferSyntaxSub const & GetTransferSyntax(SizeType i) const { return TransferSyntaxes[i]; }
  TransferSyntaxSub & GetTransferSyntax(SizeType i) { return TransferSyntaxes[i]; }
  std::vector<TransferSyntaxSub> const & GetTransferSyntaxes() const {return TransferSyntaxes; }
  SizeType GetNumberOfTransferSyntaxes() const { return TransferSyntaxes.size(); }

  void SetPresentationContextID( uint8_t id );
  uint8_t GetPresentationContextID() const;

  void Print(std::ostream &os) const;

  bool operator==(const PresentationContextRQ & pc) const
    {
    assert( TransferSyntaxes.size() == 1 ); // TODO
    assert( pc.TransferSyntaxes.size() == 1 );
    return SubItems == pc.SubItems && TransferSyntaxes == pc.TransferSyntaxes;
    }

  PresentationContextRQ(const PresentationContext & pc);

private:
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength; // len of last transfer syntax
  uint8_t /*PresentationContext*/ID;
  static const uint8_t Reserved6;
  static const uint8_t Reserved7;
  static const uint8_t Reserved8;
  /*
  This variable field shall contain the following sub-items: one Abstract
  Syntax and one or more Transfer Syntax(es). For a complete
  description of the use and encoding of these sub-items see Sections
  9.3.2.2.1 and 9.3.2.2.2.
   */
  AbstractSyntax SubItems;
  std::vector<TransferSyntaxSub> TransferSyntaxes;
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMPRESENTATIONCONTEXTRQ_H
