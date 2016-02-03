/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmStringFilter.h"
#include "gdcmGlobal.h"
#include "gdcmElement.h"
#include "gdcmByteValue.h"
#include "gdcmAttribute.h"
#include "gdcmDataSetHelper.h"

#include <string.h> // strtok

namespace gdcm
{

//-----------------------------------------------------------------------------
StringFilter::StringFilter():F(new File)
{
}
//-----------------------------------------------------------------------------
StringFilter::~StringFilter()
{
}

void StringFilter::SetDicts(const Dicts &dicts)
{
  (void)dicts;
  assert(0); // FIXME
}

std::string StringFilter::ToString(const Tag& t) const
{
  return ToStringPair(t).second;
}

std::string StringFilter::ToString(const DataElement& de) const
{
  return ToStringPair(de).second;
}

/*
std::string StringFilter::ToMIME64(const Tag& t) const
{
  return ToStringPair(t).second;
          // base64 streams have to be a multiple of 4 bytes long
          int encodedLengthEstimate = 2 * bv->GetLength();
          encodedLengthEstimate = ((encodedLengthEstimate / 4) + 1) * 4;

          char *bin = new char[encodedLengthEstimate];
          unsigned int encodedLengthActual = static_cast<unsigned int>(
            itksysBase64_Encode(
              (const unsigned char *) bv->GetPointer(),
              static_cast< unsigned long>( bv->GetLength() ),
              (unsigned char *) bin,
              static_cast< int >( 0 ) ));
          std::string encodedValue(bin, encodedLengthActual);

}
*/

#define StringFilterCase(type) \
  case VR::type: \
    { \
      Element<VR::type,VM::VM1_n> el; \
      if( !de.IsEmpty() ) { \
      el.Set( de.GetValue() ); \
      if( el.GetLength() ) { \
      os << el.GetValue(); \
      for(unsigned int i = 1; i < el.GetLength(); ++i) os << "\\" << el.GetValue(i); \
      retvalue = os.str(); } } \
    } break

std::pair<std::string, std::string> StringFilter::ToStringPair(const Tag& t) const
{
  if( t.GetGroup() == 0x2 )
    {
    const FileMetaInformation &header = GetFile().GetHeader();
    return ToStringPair(t, header);
    }
  else
    {
    const DataSet &ds = GetFile().GetDataSet();
    return ToStringPair(t, ds);
    }
}

std::pair<std::string, std::string> StringFilter::ToStringPair(const DataElement& de) const
{
  const Tag & t = de.GetTag();
  if( t.GetGroup() == 0x2 )
    {
    const FileMetaInformation &header = GetFile().GetHeader();
    return ToStringPairInternal(de, header);
    }
  else
    {
    const DataSet &ds = GetFile().GetDataSet();
    return ToStringPairInternal(de, ds);
    }
}

bool StringFilter::ExecuteQuery(std::string const & query_const, std::string &value ) const
{
//  if( t.GetGroup() == 0x2 )
//    {
//    const FileMetaInformation &header = GetFile().GetHeader();
//    return ToStringPair(query_const, header);
//    }
//  else
    {
    const DataSet &ds = GetFile().GetDataSet();
    return ExecuteQuery(query_const, ds, value);
    }
}

bool StringFilter::ExecuteQuery(std::string const & query_const,
  DataSet const &ds, std::string &retvalue ) const
{
  //std::pair<std::string, std::string> ret;
  static Global &g = Global::GetInstance();
  static const Dicts &dicts = g.GetDicts();
  static const Dict &pubdict = dicts.GetPublicDict();

  char *query = strdup( query_const.c_str() );
  const char delim[] = "/";
  const char subdelim[] = "[]@='";

  char *str1, *str2, *token, *subtoken;
  char *saveptr1, *saveptr2;
  int j;

  //bool dicomnativemodel = false;//unused
  const DataSet *curds = NULL;
  const DataElement *curde = NULL;
  Tag t;
  int state = 0;
  SmartPointer<SequenceOfItems> sqi;
  for (j = 1, str1 = query; state >= 0 ; j++, str1 = NULL)
    {
    token = System::StrTokR(str1, delim, &saveptr1);

    if (token == NULL)
      break;
    //printf("%d: %s\n", j, token);

    std::vector< std::string > subtokens;
    for (str2 = token; ; str2 = NULL)
      {
      subtoken = System::StrTokR(str2, subdelim, &saveptr2);
      if (subtoken == NULL)
        break;
      //printf(" --> %s\n", subtoken);
      subtokens.push_back( subtoken );
      }
    if( subtokens[0] == "DicomNativeModel" )
      {
      // move to next state
      assert( state == 0 );
      state = 1;
      curds = &ds;
      }
    else if( subtokens[0] == "DicomAttribute" )
      {
      if( state != 1 )
        {
        state = -1;
        break;
        }
      assert( subtokens[1] == "keyword" );
      const char *k = subtokens[2].c_str();
      /*const DictEntry &dictentry = */pubdict.GetDictEntryByKeyword(k, t);
      if( !curds->FindDataElement( t ) )
        {
        state = -1;
        break;
        }
      curde = &curds->GetDataElement( t );
      }
    else if( subtokens[0] == "Item" )
      {
      assert( state == 1 );
      assert( curde );
      assert( subtokens[1] == "number" );
      sqi = curde->GetValueAsSQ();
      if( !sqi )
        {
        state = -1;
        break;
        }
      Item const &item = sqi->GetItem( atoi( subtokens[2].c_str() ) );
      curds = &item.GetNestedDataSet();
      }
    else if( subtokens[0] == "Value" )
      {
      assert( state == 1 );
      // move to next state
      state = 2;
      assert( subtokens[1] == "number" );
#if !defined(NDEBUG)
      const ByteValue * const bv = curde->GetByteValue(); (void)bv;
      assert( bv );
      //bv->Print( std::cout << std::endl );
#endif
      }
    else
      {
      assert( subtokens.size() );
      gdcmDebugMacro( "Unhandled token: " << subtokens[0] );
      state = -1;
      }
    }
  if( state != 2 )
    {
    return false;
    }
  free( query );

  const DataElement &de = *curde;

  const DictEntry &entry = pubdict.GetDictEntry(de.GetTag());

  const VR &vr_read = de.GetVR();
  const VR &vr_dict = entry.GetVR();

  if( vr_dict == VR::INVALID )
    {
    // FIXME This is a public element we do not support...
    return false;
    }

  VR vr;
  // always prefer the vr from the file:
  if( vr_read == VR::INVALID )
    {
    vr = vr_dict;
    }
  else if ( vr_read == VR::UN && vr_dict != VR::INVALID ) // File is explicit, but still prefer vr from dict when UN
    {
    vr = vr_dict;
    }
  else // cool the file is Explicit !
    {
    vr = vr_read;
    }
  if( vr.IsDual() ) // This mean vr was read from a dict entry:
    {
    vr = DataSetHelper::ComputeVR(*F,ds, t);
    }

  if( vr == VR::UN )
    {
    // this element is not known...
    return false;
    }

  assert( vr != VR::UN && vr != VR::INVALID );
  //ret.first = entry.GetName();
  if( VR::IsASCII( vr ) )
    {
    assert( vr & VR::VRASCII );
    const ByteValue *bv = de.GetByteValue();
    if( de.GetVL() )
      {
      assert( bv /*|| bv->IsEmpty()*/ );
      retvalue = std::string( bv->GetPointer(), bv->GetLength() );
      // Let's remove any trailing \0 :
      retvalue.resize( std::min( retvalue.size(), strlen( retvalue.c_str() ) ) ); // strlen is garantee to be lower or equal to ::size()
      }
    else
      {
      //assert( bv == NULL );
      retvalue = ""; // ??
      }
    }
  else
    {
    assert( vr & VR::VRBINARY );
    const ByteValue *bv = de.GetByteValue();
    if( bv )
      {
      //VM::VMType vm = entry.GetVM();//!!mmr-- can I remove this, or will it mess with the stream?
      //assert( vm == VM::VM1 );
      if( vr.IsDual() ) // This mean vr was read from a dict entry:
        {
        vr = DataSetHelper::ComputeVR(GetFile(),ds, t);
        }
      std::ostringstream os;
      switch(vr)
        {
        StringFilterCase(AT);
        StringFilterCase(FL);
        StringFilterCase(FD);
        //StringFilterCase(OB);
        StringFilterCase(OF);
        //StringFilterCase(OW);
        StringFilterCase(SL);
        //StringFilterCase(SQ);
        StringFilterCase(SS);
        StringFilterCase(UL);
        //StringFilterCase(UN);
        StringFilterCase(US);
        StringFilterCase(UT);
      case VR::UN:
      case VR::US_SS:
        assert(0);
        break;
      case VR::OB:
      case VR::OW:
      case VR::OB_OW:
      case VR::SQ:
        gdcmWarningMacro( "Unhandled: " << vr << " for tag " << de.GetTag() );
        retvalue = "";
        break;
      default:
        assert(0);
        break;
        }
      }
    }
  return true;
}

std::pair<std::string, std::string> StringFilter::ToStringPair(const Tag& t, DataSet const &ds) const
{
  std::pair<std::string, std::string> ret;
  if( !ds.FindDataElement(t) )
    {
    gdcmDebugMacro( "DataSet does not contains tag:" );
    return ret;
    }
  const DataElement &de = ds.GetDataElement( t );
  ret = ToStringPairInternal( de, ds );
  return ret;
}

std::pair<std::string, std::string> StringFilter::ToStringPairInternal(const DataElement& de, DataSet const &ds) const
{
  std::pair<std::string, std::string> ret;
  const Global &g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();
  if( ds.IsEmpty() )
    {
    gdcmDebugMacro( "DataSet is empty or does not contains tag:" );
    return ret;
    }
  //assert( de.GetTag().IsPublic() );
  std::string strowner;
  const char *owner = 0;
  const Tag &t = de.GetTag();
  if( t.IsPrivate() && !t.IsPrivateCreator() )
    {
    strowner = ds.GetPrivateCreator(t);
    owner = strowner.c_str();
    }

  const DictEntry &entry = dicts.GetDictEntry(de.GetTag(), owner);

  const VR &vr_read = de.GetVR();
  const VR &vr_dict = entry.GetVR();

  if( vr_dict == VR::INVALID )
    {
    // FIXME This is a public element we do not support...
    return ret;
    }

  VR vr;
  // always prefer the vr from the file:
  if( vr_read == VR::INVALID )
    {
    vr = vr_dict;
    }
  else if ( vr_read == VR::UN && vr_dict != VR::INVALID ) // File is explicit, but still prefer vr from dict when UN
    {
    vr = vr_dict;
    }
  else // cool the file is Explicit !
    {
    vr = vr_read;
    }
  if( vr.IsDual() ) // This mean vr was read from a dict entry:
    {
    vr = DataSetHelper::ComputeVR(*F,ds, t);
    }

  if( vr == VR::UN )
    {
    // this element is not known...
    return ret;
    }

  assert( vr != VR::UN && vr != VR::INVALID );
  //std::cerr << "Found " << vr << " for " << de.GetTag() << std::endl;
  ret.first = entry.GetName();
  if( VR::IsASCII( vr ) )
    {
    assert( vr & VR::VRASCII );
    const ByteValue *bv = de.GetByteValue();
    if( de.GetVL() )
      {
      assert( bv /*|| bv->IsEmpty()*/ );
      ret.second = std::string( bv->GetPointer(), bv->GetLength() );
      // Let's remove any trailing \0 :
      ret.second.resize( std::min( ret.second.size(), strlen( ret.second.c_str() ) ) ); // strlen is garantee to be lower or equal to ::size()
      }
    else
      {
      //assert( bv == NULL );
      ret.second = ""; // ??
      }
    }
  else
    {
    assert( vr & VR::VRBINARY );
    const ByteValue *bv = de.GetByteValue();
    if( bv )
      {
      //VM::VMType vm = entry.GetVM();//!!mmr-- can I remove this, or will it mess with the stream?
      //assert( vm == VM::VM1 );
      if( vr.IsDual() ) // This mean vr was read from a dict entry:
        {
        vr = DataSetHelper::ComputeVR(GetFile(),ds, t);
        }
      std::ostringstream os;
      std::string retvalue;
      switch(vr)
        {
        StringFilterCase(AT);
        StringFilterCase(FL);
        StringFilterCase(FD);
        //StringFilterCase(OB);
        StringFilterCase(OF);
        //StringFilterCase(OW);
        StringFilterCase(SL);
        //StringFilterCase(SQ);
        StringFilterCase(SS);
        StringFilterCase(UL);
        //StringFilterCase(UN);
        StringFilterCase(US);
        StringFilterCase(UT);
      case VR::UN:
      case VR::US_SS:
        assert(0);
        break;
      case VR::OB:
      case VR::OW:
      case VR::OB_OW:
      case VR::SQ:
        gdcmWarningMacro( "Unhandled: " << vr << " for tag " << de.GetTag() );
        ret.second = "";
        break;
      default:
        assert(0);
        break;
        }
        ret.second = retvalue;
      }
    }
  return ret;
}

#if !defined(GDCM_LEGACY_REMOVE)
std::string StringFilter::FromString(const Tag&t, const char * value, VL const & vl)
{
  (void)t;
  (void)value;
  (void)vl;
  assert(0 && "TODO");
  return "";
}
#endif

#define FromStringFilterCase(type) \
  case VR::type: \
      { \
      Element<VR::type,VM::VM1_n> el; \
      /* el.ReadComputeLength( is ); */ \
      el.SetLength( vl );  \
       for(unsigned int i = 0; i < vm.GetLength(); ++i)  \
        { \
        if(i) is.get(); \
        is >> el.GetValue(i);  \
        } \
      el.Write(os); \
      } \
    break

static inline size_t count_backslash(const char *s, size_t len)
{
  assert( s );
  size_t c = 0;
  for(size_t i = 0; i < len; ++i, ++s)
    {
    if( *s == '\\' )
      {
      ++c;
      }
    }
  return c;
}

std::string StringFilter::FromString(const Tag&t, const char * value, size_t len)
{
  if( !value || !len ) return "";
  const Global &g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();
  std::string strowner;
  const char *owner = 0;
  const DataSet &ds = GetFile().GetDataSet();
  if( t.IsPrivate() && !t.IsPrivateCreator() )
    {
    strowner = ds.GetPrivateCreator(t);
    owner = strowner.c_str();
    }

  const DictEntry &entry = dicts.GetDictEntry(t, owner);
  const VM &vm = entry.GetVM();
  //const VR &vr = entry.GetVR();
  const DataElement &de = ds.GetDataElement( t );
  const VR &vr_read = de.GetVR();
  const VR &vr_dict = entry.GetVR();

  VR vr;
  // always prefer the vr from the file:
  if( vr_read == VR::INVALID )
    {
    vr = vr_dict;
    }
  else if ( vr_read == VR::UN && vr_dict != VR::INVALID ) // File is explicit, but still prefer vr from dict when UN
    {
    vr = vr_dict;
    }
  else // cool the file is Explicit !
    {
    vr = vr_read;
    }
  if( vr.IsDual() ) // This mean vr was read from a dict entry:
    {
    vr = DataSetHelper::ComputeVR(*F,ds, t);
    }

  if( vr == VR::UN )
    {
    // this element is not known...
    //return ret;
    }

  std::string s(value,value+len);
  if( VR::IsASCII( vr ) )
    {
    return s;
    }
  VL::Type castLen = (VL::Type)len;
  VL::Type count = VM::GetNumberOfElementsFromArray(value, castLen);
  VL vl = vm.GetLength() * vr.GetSizeof();
  if( vm.GetLength() == 0 )
    {
    // VM1_n
    vl = count * vr.GetSizeof();
#if !defined(NDEBUG)
    VM check  = VM::GetVMTypeFromLength(count, 1);
    assert( vm.Compatible( check ) );
#endif
    }

  std::istringstream is;
  is.str( s );
  std::ostringstream os;
  switch(vr)
    {
    FromStringFilterCase(AT);
    FromStringFilterCase(FL);
    FromStringFilterCase(FD);
    //FromStringFilterCase(OB);
    FromStringFilterCase(OF);
    //FromStringFilterCase(OW);
    FromStringFilterCase(SL);
    //FromStringFilterCase(SQ);
    FromStringFilterCase(SS);
    FromStringFilterCase(UL);
    //FromStringFilterCase(UN);
    FromStringFilterCase(US);
  default:
    gdcmErrorMacro( "Not implemented" );
    assert(0);
    }
  return os.str();
}

} // end namespace gdcm
