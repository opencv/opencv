/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPresentationContextGenerator.h"

#include "gdcmReader.h"
#include "gdcmTransferSyntax.h"

#include <algorithm> // std::find

namespace gdcm
{
std::string PresentationContextGenerator::DefaultTransferSyntax = "1.2.840.10008.1.2";

const char *PresentationContextGenerator::GetDefaultTransferSyntax() const
{
  return DefaultTransferSyntax.c_str();
}

void PresentationContextGenerator::SetDefaultTransferSyntax( const TransferSyntax &ts )
{
  DefaultTransferSyntax = ts.GetString();
}

PresentationContextGenerator::PresentationContextGenerator() {}

//bool PresentationContextGenerator::GenerateFromQuery(BaseRootQuery *query)
//{
//}

bool PresentationContextGenerator::GenerateFromUID(UIDs::TSName asname)
{
  PresContext.clear();

  const char *asnamestr = UIDs::GetUIDString( asname );
  const char *tsnamestr = GetDefaultTransferSyntax();

  // There is a special case for MOVE operations. Need to have alternate FIND operations
  if( asname == UIDs::PatientRootQueryRetrieveInformationModelMOVE )
    {
    const char *asnamestr0 = UIDs::GetUIDString(
      UIDs::PatientRootQueryRetrieveInformationModelFIND );
    AddPresentationContext( asnamestr0, tsnamestr );
    }
  else if( asname == UIDs::StudyRootQueryRetrieveInformationModelMOVE )
    {
    const char *asnamestr0 = UIDs::GetUIDString(
      UIDs::StudyRootQueryRetrieveInformationModelFIND );
    AddPresentationContext( asnamestr0, tsnamestr );
    }
  // Always set the *MOVE after the *FIND one
  AddPresentationContext( asnamestr, tsnamestr );

  return true;
}

bool PresentationContextGenerator::GenerateFromFilenames(const Directory::FilenamesType &filenames)
{
  PresContext.clear();

  // By design GDCM C-STORE implementation only setup the association for any dataset we are
  // about to send. This is therefore very important to gather all possible SOP Class
  // we are about to send otherwise the other end will simply disconnect us
  // this imply that C-STORE will refuse any DataSet without SOP Clas or SOP Instances
  Tag tsuid(0x2,0x10);
  Tag mediasopclass(0x2,0x2);
  Tag sopclass(0x8,0x16);
#if 0
  Scanner sc;
  sc.AddTag( tsuid );
  sc.AddTag( mediasopclass );
  sc.AddTag( sopclass );
  if( !sc.Scan( filenames ) )
    {
    gdcmErrorMacro( "Could not scan filenames" );
    return 1;
    }
#endif

  Directory::FilenamesType::const_iterator file = filenames.begin();
  std::set<Tag> skiptags;
  for(; file != filenames.end(); ++file)
    {
    // I cannot use gdcm::Scanner as I need the TS of the file. When the file
    // only contains the DataSet I cannot know if this is Explicit or Implicit
    // Instead re-use the lower level bricks of gdcm::Scanner here:
    const char *fn = file->c_str();
    Reader reader;
    reader.SetFileName( fn );
    // NOTE: There is a small overhead right here: what if we are sending Deflated
    // TS encoded file. We should not need to read up to tag 8,16 ...
    if( reader.ReadUpToTag( sopclass, skiptags ) )
      {
      DataSet const & ds = reader.GetFile().GetDataSet();
      if( ds.FindDataElement( sopclass ) )
        {
        // by design gdcmscu will not send/retrieve a dataset that cannot be read
        // this should not be too restricitive
        const TransferSyntax &fmits = reader.GetFile().GetHeader().GetDataSetTransferSyntax();
        //const char *tsuidvalue = sc.GetValue(fn, tsuid);
        const char *tsuidvalue = fmits.GetString();
        //const char *sopclassvalue = sc.GetValue(fn, sopclass );

        const DataElement &tsde = ds.GetDataElement( sopclass );
        //const char *sopclassvalue = sc.GetValue(fn, sopclass );
        // Passing pointer directly. We do not try to analyze what Media Storage
        // it is. We should be able to support to send/receive unknwon media storage
        const ByteValue *bv = tsde.GetByteValue();
        std::string buffer( bv->GetPointer(), bv->GetLength() );
        const char *sopclassvalue = buffer.c_str();

        AddPresentationContext( sopclassvalue, tsuidvalue );
        }
      }
    else
      {
      gdcmErrorMacro( "Could not read: " << fn );
      }
    }

  return true;
}

bool PresentationContextGenerator::AddPresentationContext( const char *as, const char * ts )
{
  // \precondition
  assert( as );
  assert( ts );

  SizeType n = PresContext.size();
  PresentationContext pc;
  pc.SetAbstractSyntax( as );
  SizeType idn = 2*n + 1;
  assert( idn <= std::numeric_limits<uint8_t>::max() );
  pc.SetPresentationContextID( (uint8_t)idn );
  pc.AddTransferSyntax( ts );

  PresentationContextArrayType::const_iterator it =
    std::find( PresContext.begin(), PresContext.end(), pc );

  // default mode it to only append when pc is not present already:
  // warning dcmtk 3.5.4 will segfault if no PresentationContext is found
  // (fixed in 3.6.0)
  if( it == PresContext.end() )
    {
    PresContext.push_back( pc );
    }

  return true;
}

void PresentationContextGenerator::SetMergeModeToAbstractSyntax()
{
}

void PresentationContextGenerator::SetMergeModeToTransferSyntax()
{
  assert( 0 && "TODO" );
}

} // end namespace gdcm
