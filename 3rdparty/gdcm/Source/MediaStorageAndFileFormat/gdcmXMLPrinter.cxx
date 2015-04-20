/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "gdcmXMLPrinter.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmDict.h"
#include "gdcmDicts.h"
#include "gdcmGroupDict.h"
#include "gdcmVR.h"
#include "gdcmVM.h"
#include "gdcmElement.h"
#include "gdcmGlobal.h"
#include "gdcmAttribute.h"
#include "gdcmDataSetHelper.h"
#include "gdcmUUIDGenerator.h"

#include "gdcmDataSet.h"

#include <typeinfo>


namespace gdcm
{
//-----------------------------------------------------------------------------
XMLPrinter::XMLPrinter():PrintStyle(XMLPrinter::OnlyUUID),F(0)
{
}

//-----------------------------------------------------------------------------
XMLPrinter::~XMLPrinter()
{
}

// Carried forward from Printer Class
// SIEMENS_GBS_III-16-ACR_NEMA_1.acr is a tough kid: 0009,1131 is supposed to be VR::UL, but
// there are only two bytes...

VR XMLPrinter::PrintDataElement(std::ostream &os, const Dicts &dicts, const DataSet & ds,
  const DataElement &de, const TransferSyntax & ts)
{

  const ByteValue *bv = de.GetByteValue();
  const SequenceOfItems *sqi = 0;
  const Value &value = de.GetValue();
  const SequenceOfFragments *sqf = de.GetSequenceOfFragments();

  std::string strowner;
  const char *owner = 0;
  const Tag& t = de.GetTag();
  UUIDGenerator UIDgen;

  if( t.IsPrivate() && !t.IsPrivateCreator() )
    {
    strowner = ds.GetPrivateCreator(t);
    owner = strowner.c_str();
    }
  const DictEntry &entry = dicts.GetDictEntry(t,owner);
  const VR &vr = entry.GetVR();
  const VM &vm = entry.GetVM();
  (void)vm;
  const char *name = entry.GetKeyword();
  bool retired = entry.GetRetired();

  const VR &vr_read = de.GetVR();
  const VL &vl_read = de.GetVL();

  //Printing Tag

  
  os << " tag = \"" << std::uppercase << std::hex << std::setw(4) << std::setfill('0') <<
      t.GetGroup() <<  std::setw(4) << t.GetElement() << std::nouppercase <<"\"" << std::dec;

  //Printing Private Creator
  if( owner && *owner )
    {
    os << " privateCreator = \"" << owner << "\" ";
    }

  VR refvr;

  // Prefer the vr from the file:

  if( vr_read == VR::INVALID )
    {
    refvr = vr;
    }
  else if ( vr_read == VR::UN && vr != VR::INVALID ) // File is explicit, but still prefer vr from dict when UN
    {
    refvr = vr;
    }
  else // The file is Explicit !
    {
    refvr = vr_read;
    }

  if( refvr.IsDual() ) // This means vr was read from a dict entry:
    {
    refvr = DataSetHelper::ComputeVR(*F,ds, t);
    }

  //as DataSetHelper would have been called
  assert( refvr != VR::US_SS );
  assert( refvr != VR::OB_OW );

  if( dynamic_cast<const SequenceOfItems*>( &value ) )
    {
    sqi = de.GetValueAsSQ();
    refvr = VR::SQ;
    assert( refvr == VR::SQ );
    }
#if 0
  else if( vr == VR::SQ && vr_read != VR::SQ )
    {
    sqi = de.GetValueAsSQ();
    refvr = VR::SQ;
    assert( refvr == VR::SQ );
    }
#endif

  if( (vr_read == VR::INVALID || vr_read == VR::UN ) && vl_read.IsUndefined() )
    {
    assert( refvr == VR::SQ );
    }

//  if( vr_read == VR::SQ || vr_read == VR::UN )
//    {
//    sqi = de.GetValueAsSQ();
//    }
  if( vr != VR::INVALID && (!vr.Compatible( vr_read ) || vr_read == VR::INVALID || vr_read == VR::UN ) )
    {
    assert( vr != VR::INVALID );

    /*
    No need as we will save only the VR to which it is stored by GDCM in the XML file
    
    if( vr == VR::US_SS || vr == VR::OB_OW )
      {
      os << "(" << vr << " => " << refvr << ") ";
      }
    else
      {
      os << "(" << vr << ") ";
      }
     */ 
    
    }
  else if( sqi /*de.GetSequenceOfItems()*/ && refvr == VR::INVALID )
    {
    // when vr == VR::INVALID and vr_read is also VR::INVALID, we have a seldom case where we can guess
    // the vr
    // eg. CD1/647662/647663/6471066 has a SQ at (2001,9000)

    assert( refvr == VR::INVALID );
    refvr = VR::SQ;
    }

  if(refvr == VR::INVALID)
    refvr = VR::UN;

  // Printing the VR -- Value Representation
  os << " vr = \"" << refvr << "\" ";

  // Add the keyword attribute :

  if( t.IsPublic())
    {
    if( name && *name )
      {
      os <<"keyword = \"";

      /*  No owner */
      if( t.IsPrivate() && (owner == 0 || *owner == 0 ) && !t.IsPrivateCreator() )
        {
        //os << name;
        //os = PrintXML_char(os,name);
        }
      /* retired element */
      else if( retired )
        {
        assert( t.IsPublic() || t.GetElement() == 0x0 ); // Is there such thing as private and retired element ?
        //os << name;
        //os = PrintXML_char(os,name);
        }
      /* Public element */
      else
        {
        //os << name;
        //os = PrintXML_char(os,name);
        }

      char c;
      for (; (*name)!='\0'; name++)
        {
        c = *name;
        if(c == '&')
          os << "&amp;";
        else if(c == '<')
          os << "&lt;";
        else if(c == '>')
          os << "&gt;";
        else if(c == '\'')
          os << "&apos;";
        else if(c == '\"')
          os << "&quot;";
        else
          os << c;
        }
      os << "\"";
      }
    else
      {
      if( t.IsPublic() )
        {
        gdcmWarningMacro( "An unknown public element.");
        }
      //    os << ""; // Special keyword
      }
    }
  os << ">\n";

#define StringFilterCase(type) \
  case VR::type: \
    { \
      Element<VR::type,VM::VM1_n> el; \
      if( !de.IsEmpty() ) { \
      el.Set( de.GetValue() ); \
      if( el.GetLength() ) { \
      os << "<Value number = \"1\" >" ;os << "" << el.GetValue();os << "</Value>\n"; \
      const uint32_t l = (uint32_t)el.GetLength(); \
      for(uint32_t i = 1; i < l; ++i) \
      { \
      os << "<Value number = \"" << (i+1) << "\" >" ;\
      os << el.GetValue(i);os << "</Value>\n";} \
      } \
      else { if( de.IsEmpty() ) \
                 {} } } \
      else { assert( de.IsEmpty()); } \
    } break


  // Print Value now:

  //Handle PN first, acc. to Standard
  if(refvr == VR::PN)
    {
    if( bv )
      {
      bv->PrintPNXML(os);    //new function to print each value in new child tag
      }
    else
      {
      assert( de.IsEmpty() );
      }
    }
  else if( refvr & VR::VRASCII )
    {
    //assert( !sqi && !sqf);
    assert(!sqi);
    if( bv )
      {
      bv->PrintASCIIXML(os);    //new function to print each value in new child tag
      }
    else
      {
      assert( de.IsEmpty() );
      }
    }
  else
    {
    assert( refvr & VR::VRBINARY || (vr == VR::INVALID && refvr == VR::INVALID) );
    std::string s;
    switch(refvr)
      {
      StringFilterCase(AT);
      StringFilterCase(FL);
      StringFilterCase(FD);
      StringFilterCase(OF);
      StringFilterCase(SL);
      StringFilterCase(SS);
      StringFilterCase(UL);
      StringFilterCase(US);
    case VR::OB:
    case VR::OW:
    case VR::OB_OW:
    case VR::UN:
    case VR::US_SS_OW: 
        {
        if ( bv )
          {
          if(PrintStyle)
            {
            bv->PrintHexXML(os);
            }
          else
            {
            if(bv->GetLength())
              {
              const char *suid = UIDgen.Generate();
              os << "<BulkData uuid = \""<<
                suid << "\" />\n";
              HandleBulkData( suid, ts, bv->GetPointer(), bv->GetLength() );
              }
            }
          }
        else if ( sqf )
          {
          assert( t == Tag(0x7fe0,0x0010) );
          }
        else if ( sqi )
          {
          gdcmErrorMacro( "Should not happen: VR=UN but contains a SQ" );
          }
        else
          {
          assert( !sqi && !sqf );
          assert( de.IsEmpty() );
          }
        }
      break;
    case VR::US_SS:
      assert( refvr != VR::US_SS );
      break;
    case VR::SQ://The below info need not be printed into the XML infoset acc. to the standard
      if( !sqi && !de.IsEmpty() && de.GetValue().GetLength() )
        {
        }
      else
        {
        if( vl_read.IsUndefined() )
          {
          //os << "(Sequence with undefined length)";
          }
        else
          {
          //os << "(Sequence with defined length)";
          }
        }
      break;
    case VR::INVALID:
      if( bv )
        {
        if(PrintStyle)
          bv->PrintHexXML(os);
        else
          {
          if(bv->GetLength())
            {
            const char *suid = UIDgen.Generate();
            os << "<BulkData uuid = \""<<
              suid << "\" />\n";
            HandleBulkData( suid, ts, bv->GetPointer(), bv->GetLength() );
            }
          }
        }
      else
        {
        assert( !sqi && !sqf );
        assert( de.IsEmpty() );
        }
      break;
    default:
      assert(0 && "No Match! Impossible!!");
      break;
      }
    }

  //os << "\n";
  return refvr;
}

void XMLPrinter::PrintSQ(const SequenceOfItems *sqi, const TransferSyntax & ts, std::ostream & os)
{
  if( !sqi ) return;
  
  int noItems = 1;

  SequenceOfItems::ItemVector::const_iterator it = sqi->Items.begin();
  for(; it != sqi->Items.end(); ++it)
    {
    const Item &item = *it;
    const DataSet &ds = item.GetNestedDataSet();
    //const DataElement &deitem = item;
    /*
    os << "<DicomAttribute  tag = \"";
    os << std::hex << std::setw(4) << std::setfill('0') <<
      deitem.GetTag().GetGroup() <<  std::setw(4) << ((uint16_t)(deitem.GetTag().GetElement() << 8) >> 8) << "\" ";
    os << " VR = \"UN\"  keyword = ";     

    if( deitem.GetVL().IsUndefined() )
      {
      os << "\"ItemWithUndefinedLength\"";
      }
    else
      {
      os << "\"ItemWithDefinedLength\"";
      }
    os << ">\n";
    */
    os << "<Item number = \"" << noItems++ << "\">\n";
    PrintDataSet(ds, ts, os);
    /*
    if( deitem.GetVL().IsUndefined() )
      {
            os << "<DicomAttribute    tag = \"fffee00d\"  VR = \"UN\" keyword = \"ItemDelimitationItem\"/>\n";
      }
    os << "</DicomAttribute>\n\n";  
    */
    os << "</Item>\n";
    }
  /*
  if( sqi->GetLength().IsUndefined() )
    {
        os << "<DicomAttribute    tag = \"fffee0dd\"  VR = \"UN\" keyword = \"SequenceDelimitationItem\"/>\n";
    }
  */  
}

void XMLPrinter::PrintDataSet(const DataSet &ds, const TransferSyntax & ts, std::ostream &os)
{
  const Global& g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();
  const Dict &d = dicts.GetPublicDict(); (void)d;

  DataSet::ConstIterator it = ds.Begin();
  UUIDGenerator UIDgen;
  
  for( ; it != ds.End(); ++it )
    {
    const DataElement &de = *it;
    
    const SequenceOfFragments *sqf = de.GetSequenceOfFragments();
    
    os << "<DicomAttribute  " ;
    VR refvr = PrintDataElement(os, dicts, ds, de, ts);

    if( refvr == VR::SQ /*|| sqi*/ )
      {
      SmartPointer<SequenceOfItems> sqi2 = de.GetValueAsSQ();
      PrintSQ(sqi2, ts, os);
      }
    else if ( sqf )
      {
      /*I have appended all fragments into one by calling the GetBuffer method in 
      gdcmSequenceOfFragments which does not write the Table to the buffer.
      It is slightly buggy as the size returnes includes that of the table.
      Should I get the Table size and subtract it?
      Or should I append the table as well in the BulkData??
      */
      unsigned long size = sqf->ComputeByteLength();
      char *bulkData = new char [size];
      if(sqf->GetBuffer(bulkData, size))
      	{
      	if(size)
      		{
      		const char *suid = UIDgen.Generate();
        	os << "<BulkData uuid = \""<<
        				 suid << "\" />\n";
        	HandleBulkData( suid, ts, bulkData, size);
      		}
      	}
      /*      
      const BasicOffsetTable & table = sqf->GetTable();
      const ByteValue *bv = table.GetByteValue();
      
      if(bv->GetLength())
      	{
        const char *suid = UIDgen.Generate();
        os << "<BulkData uuid = \""<<
        			 suid << "\" />\n";
        HandleBulkData( suid, ts, bv->GetPointer(), bv->GetLength() );
        }
      
      unsigned int numfrag = sqf->GetNumberOfFragments();
      for(unsigned int i = 0; i < numfrag; i++)
        {        
        const Fragment& frag = sqf->GetFragment(i);
        const ByteValue *bv = frag.GetByteValue();
        if(bv->GetLength())
      		{
        	const char *suid = UIDgen.Generate();
        	os << "<BulkData uuid = \""<<
        				 suid << "\" />\n";
        	HandleBulkData( suid, ts, bv->GetPointer(), bv->GetLength() );
        	}        
        }
      */
      }
    else
      {
      // This is a byte value, so it should have been already treated.
      }
    os << "</DicomAttribute>\n";
    }
}



/*------------------------------------------------------------------------------------------------------------------------------------------------*/

void XMLPrinter::Print(std::ostream& os)
{
  /* XML Meta Info */
  const Tag CharacterEncoding(0x0008,0x0005);

  const DataSet &ds = F->GetDataSet();
  const FileMetaInformation &header = F->GetHeader();
  const TransferSyntax &ts = header.GetDataSetTransferSyntax();

  os << "<?xml version=\"1.0\" encoding=\"";
  if(ds.FindDataElement(CharacterEncoding))
    {
    const DataElement &de = ds.GetDataElement(CharacterEncoding);
    if( !de.IsEmpty() )
      {
      Attribute<0x8,0x5> at;
      at.SetFromDataElement( de );
      const char* EncodingFromFile = at.GetValue(0);
      if (!strcmp(EncodingFromFile,"ISO_IR 6"))
        os << "UTF-8";
      else if (!strcmp(EncodingFromFile,"ISO_IR 192"))
        os << "UTF-8";
      else if (!strcmp(EncodingFromFile,"ISO_IR 100"))
        os << "ISO-8859-1";
      else if (!strcmp(EncodingFromFile,"ISO_IR 101"))
        os << "ISO-8859-2";
      else if (!strcmp(EncodingFromFile,"ISO_IR 109"))
        os << "ISO-8859-3";
      else if (!strcmp(EncodingFromFile,"ISO_IR 110"))
        os << "ISO-8859-4";
      else if (!strcmp(EncodingFromFile,"ISO_IR 148"))
        os << "ISO-8859-9";
      else if (!strcmp(EncodingFromFile,"ISO_IR 144"))
        os << "ISO-8859-5";
      else if (!strcmp(EncodingFromFile,"ISO_IR 127"))
        os << "ISO-8859-6";
      else if (!strcmp(EncodingFromFile,"ISO_IR 126"))
        os << "ISO-8859-7";
      else if (!strcmp(EncodingFromFile,"ISO_IR 138"))
        os << "ISO-8859-8";
      else
        os << "UTF-8";
      os << "\"?>\n";
      }
    else
      {
      os << "UTF-8\"?>\n\n";
      }
    }
  else
    {
    os << "UTF-8\"?>\n\n";
    }
  os << "<NativeDicomModel xmlns=\"http://dicom.nema.org/PS3.19/models/NativeDICOM\">\n";

  PrintDataSet(ds, ts, os);

  os << "</NativeDicomModel>";
}

// Drop BulkData by default.
// Application programmer can override this mechanism.
void XMLPrinter::HandleBulkData(const char *uuid, const TransferSyntax & ts,
  const char *bulkdata, size_t bulklen)
{
  (void)ts;
  (void)uuid;
  (void)bulkdata;
  (void)bulklen;
}

}//end namespace gdcm
