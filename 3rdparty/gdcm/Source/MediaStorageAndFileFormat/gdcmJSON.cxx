/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmJSON.h"
#include "gdcmGlobal.h"
#include "gdcmDicts.h"
#include "gdcmBase64.h"
#include "gdcmSystem.h"

#ifdef GDCM_USE_SYSTEM_JSON
#include <json.h>
#endif

/*
 * Implementation is done based on Sup166, which may change in the future.
 */

// TODO: CP 246 / VR=SQ + Sequence: !

// Clarification needed:
//  "00081070":{
//    "Tag":"00081070",
//    "VR":"PN",
//    "Value":[
//      null
//    ]
//  },


// Need to pay attention that:
// \ISO 2022 IR 13\ISO 2022 IR 87
// encodes to:
// [ null, "ISO 2022 IR 13", "ISO 2022 IR 87" ]
// or ?
// [ "", "ISO 2022 IR 13", "ISO 2022 IR 87" ]

// TODO:
// DS/IS should be encoded as number

// Question, does F.2.5 DICOM JSON Model Null Values
// imply:
// "Sequence": [ null ] ?

// does not make any sense to send Group Length...

// Clarification needed, specs says IS should be a Number but example uses
// String

namespace gdcm
{

#ifdef GDCM_USE_SYSTEM_JSON
static inline void wstrim(std::string& str)
{
  str.erase(0, str.find_first_not_of(' '));
  str.erase(str.find_last_not_of(' ')+1);
}

static inline bool CanContainBackslash( const VR::VRType vrtype )
{
  assert( VR::IsASCII( vrtype ) );
  // PS 3.5-2011 / Table 6.2-1 DICOM VALUE REPRESENTATIONS
  switch( vrtype )
    {
  case VR::AE: // ScheduledStationAETitle
    //case VR::AS: // no
    //case VR::AT: // binary
  case VR::CS: // SpecificCharacterSet
  case VR::DA: // CalibrationDate
  case VR::DS: // FrameTimeVector
  case VR::DT: // ReferencedDateTime
    //case VR::FD: // binary
    //case VR::FL:
  case VR::IS: // ReferencedFrameNumber
  case VR::LO: // OtherPatientIDs
    //case VR::LT: // VM1
    //case VR::OB: // binary
    //case VR::OD: // binary
    //case VR::OF: // binary
    //case VR::OW: // binary
  case VR::PN: // PerformingPhysicianName
  case VR::SH: // PatientTelephoneNumbers
    //case VR::SL: // binary
    //case VR::SQ: // binary
    //case VR::SS: // binary
    //case VR::ST: // VM1
  case VR::TM: // CalibrationTime
  case VR::UI: // SOPClassesInStudy
    //case VR::UL: // binary
    //case VR::UN: // binary
    //case VR::US: // binary
    //case VR::UT: // VM1
    assert( !(vrtype & VR::VR_VM1) );
    return true;
  default:
    ;
    }
  return false;
}
#endif

class JSONInternal
{
public:
  JSONInternal():PrettyPrint(false),PreferKeyword(false){}
  bool PrettyPrint;
  bool PreferKeyword;
};

JSON::JSON()
{
  Internals = new JSONInternal;
}

JSON::~JSON()
{
  delete Internals;
}

void JSON::SetPrettyPrint(bool onoff)
{
  Internals->PrettyPrint = onoff;
}
bool JSON::GetPrettyPrint() const
{
  return Internals->PrettyPrint;
}
void JSON::PrettyPrintOn()
{
  Internals->PrettyPrint = true;
}
void JSON::PrettyPrintOff()
{
  Internals->PrettyPrint = false;
}

#ifdef GDCM_USE_SYSTEM_JSON

/*
  "StudyInstanceUID": {
    "Tag": "0020000D",
    "VR": "UI",
    "Value": [ "1.2.392.200036.9116.2.2.2.1762893313.1029997326.945873" ]
    },
*/

static void DataElementToJSONArray( const VR::VRType vr, const DataElement & de, json_object *my_array )
{
  if( de.IsEmpty() )
    {
    // F.2.5 DICOM JSON Model Null Values
    if( vr == VR::PN )
      {
      json_object *my_object_comp = json_object_new_object();
      json_object_array_add(my_array, my_object_comp );
      }
    //else
    //  json_object_array_add(my_array, NULL );
    return;
    }
  // else
  assert( !de.IsEmpty() );
  const bool checkbackslash = CanContainBackslash( vr );
  const ByteValue * bv = de.GetByteValue();
  const char * value = bv->GetPointer();
  size_t len = bv->GetLength();

  if( vr == VR::UI )
    {
    const std::string strui( value, len );
    const size_t lenuid = strlen( strui.c_str() ); // trick to remove trailing \0
    json_object_array_add(my_array, json_object_new_string_len(strui.c_str(), lenuid));
    }
  else if( vr == VR::PN )
    {
    const char *str1 = value;
    // remove whitespace:
    while( str1[len-1] == ' ' )
      {
      len--;
      }
    assert( str1 );
    std::stringstream ss;
    static const char *Keys[] = {
      "Alphabetic",
      "Ideographic",
      "Phonetic",
    };
    while (1)
      {
      assert( str1 && (size_t)(str1 - value) <= len );
      const char * sep = strchr(str1, '\\');
      const size_t llen = (sep != NULL) ? (sep - str1) : (value + len - str1);
      const std::string component(str1, llen);

      const char *str2 = component.c_str();
      assert( str2 );
      const size_t len2 = component.size();
      assert( len2 == llen );

      int idx = 0;
      json_object *my_object_comp = json_object_new_object();
      while (1)
        {
        assert( str2 && (size_t)(str2 - component.c_str() ) <= len2 );
        const char * sep2 = strchr(str2, '=');
        const size_t llen2 = (sep2 != NULL) ? (sep2 - str2) : (component.c_str() + len2 - str2);
        const std::string group(str2, llen2);
        const char *thekey = Keys[idx++];

        json_object_object_add(my_object_comp, thekey,
          json_object_new_string_len( group.c_str(), group.size() ) );
        if (sep2 == NULL) break;
        str2 = sep2 + 1;
        }
      json_object_array_add(my_array, my_object_comp);
      if (sep == NULL) break;
      str1 = sep + 1;
      assert( checkbackslash );
      }
    }
  else if( vr == VR::DS || vr == VR::IS )
    {
    const char *str1 = value;
    assert( str1 );
    VRToType<VR::IS>::Type vris;
    VRToType<VR::DS>::Type vrds;
    while (1)
      {
      std::stringstream ss;
      assert( str1 && (size_t)(str1 - value) <= len );
      const char * sep = strchr(str1, '\\');
      const size_t llen = (sep != NULL) ? (sep - str1) : (value + len - str1);
      // This is complex, IS/DS should not be stored as string anymore
      switch( vr )
        {
      case VR::IS:
        ss.str( std::string(str1, llen) );
        ss >> vris;
        json_object_array_add(my_array, json_object_new_int(vris)); // json_object_new_int takes an int32_t
        break;
      case VR::DS:
        ss.str( std::string(str1, llen) );
        ss >> vrds;
        json_object_array_add(my_array, json_object_new_double(vrds));
        break;
      default:
        assert( 0 ); // programmer error
        }
      if (sep == NULL) break;
      str1 = sep + 1;
      assert( checkbackslash );
      }
    }
  else if( checkbackslash )
    {
    const char *str1 = value;
    assert( str1 );
    while (1)
      {
      assert( str1 && (size_t)(str1 - value) <= len );
      const char * sep = strchr(str1, '\\');
      const size_t llen = (sep != NULL) ? (sep - str1) : (value + len - str1);
      json_object_array_add(my_array, json_object_new_string_len(str1, llen));
      if (sep == NULL) break;
      str1 = sep + 1;
      }
    }
  else // default
    {
    json_object_array_add(my_array, json_object_new_string_len(value, len));
    }
}

// Encode from DICOM to JSON
// TODO: do I really need libjson for this task ?
// FIXME: once again everything is loaded into memory
static void ProcessNestedDataSet( const DataSet & ds, json_object *my_object, const bool preferkeyword )
{
  const Global& g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();
  const Dict &d = dicts.GetPublicDict(); (void)d;

  std::vector<char> buffer;
  for( DataSet::ConstIterator it = ds.Begin();
    it != ds.End(); ++it )
    {
    const DataElement &de = *it;
    VR::VRType vr = de.GetVR();
    const Tag& t = de.GetTag();
    if( t.IsGroupLength() ) continue; // I do not see why we should send those ?
    std::string strowner;
    const char *owner = 0;
    if( t.IsPrivate() && !t.IsPrivateCreator() )
      {
      strowner = ds.GetPrivateCreator(t);
      owner = strowner.c_str();
      }
    const DictEntry &entry = dicts.GetDictEntry(t,owner);
    const std::string & str_tag = t.PrintAsContinuousUpperCaseString();
    const bool issequence = vr == VR::SQ || de.IsUndefinedLength();
    const bool isprivatecreator = t.IsPrivateCreator();
    if( issequence ) vr = VR::SQ;
    else if( isprivatecreator ) vr = VR::LO; // always prefer VR::LO (over invalid/UN)
    else if( vr == VR::INVALID ) vr = VR::UN;
    const char * vr_str = VR::GetVRString(vr);
    assert( VR::GetVRTypeFromFile(vr_str) != VR::INVALID );

    json_object *my_object_cur;
    my_object_cur = json_object_new_object();
    json_object *my_array;
    my_array = json_object_new_array();

    //json_object_object_add(my_object_cur, "Tag",
    //  json_object_new_string( str_tag.c_str()) );
    json_object_object_add(my_object_cur, "VR",
      json_object_new_string_len( vr_str, 2 ) );
    if( owner )
      {
      json_object_object_add(my_object_cur, "PrivateCreator",
        json_object_new_string( owner ) );
      }

    if( vr == VR::SQ )
      {
      SmartPointer<SequenceOfItems> sqi;
      sqi = de.GetValueAsSQ();
      if( sqi )
        {
        int nitems = sqi->GetNumberOfItems();
        for(int i = 1; i <= nitems; ++i)
          {
          const Item &item = sqi->GetItem( i );
          const DataSet &nested = item.GetNestedDataSet();

          json_object *my_object_sq;
          my_object_sq = json_object_new_object();

          ProcessNestedDataSet( nested, my_object_sq, preferkeyword );
          json_object_array_add(my_array, my_object_sq );
          }
        }
      else if( const SequenceOfFragments *sqf = de.GetSequenceOfFragments() )
        {
        json_object_array_add(my_array, NULL ); // FIXME
        assert( 0 );
        }
      else
        {
        assert( de.IsEmpty() );
        //json_object_array_add(my_array, NULL ); // F.2.5 req ?
        }
      //json_object_object_add(my_object_cur, "Sequence", my_array );
      int numel = json_object_array_length ( my_array );
      if( numel )
        json_object_object_add(my_object_cur, "Value", my_array );
      }
    else if( VR::IsASCII( vr ) )
      {
      DataElementToJSONArray( vr, de, my_array );
      int numel = json_object_array_length ( my_array );
      if( numel )
        {
        if( vr == VR::PN )
          {
          //json_object_object_add(my_object_cur, "PersonName", my_array );
          json_object_object_add(my_object_cur, "Value", my_array );
          }
        else
          json_object_object_add(my_object_cur, "Value", my_array );
        }
      }
    else
      {
      const char *wheretostore = "Value";
      switch( vr )
        {
      case VR::FD:
          {
          // Does not work, see https://github.com/json-c/json-c/pull/59
          Element<VR::FD,VM::VM1_n> el;
          el.Set( de.GetValue() );
          int ellen = el.GetLength();
          for( int i = 0; i < ellen; ++i )
            {
            json_object_array_add(my_array, json_object_new_double( el.GetValue( i ) ));
            }
          }
        break;
      case VR::FL:
          {
          Element<VR::FL,VM::VM1_n> el;
          el.Set( de.GetValue() );
          int ellen = el.GetLength();
          for( int i = 0; i < ellen; ++i )
            {
            json_object_array_add(my_array, json_object_new_double( el.GetValue( i ) ));
            }
          }
        break;
      case VR::SS:
          {
          Element<VR::SS,VM::VM1_n> el;
          el.Set( de.GetValue() );
          int ellen = el.GetLength();
          for( int i = 0; i < ellen; ++i )
            {
            json_object_array_add(my_array, json_object_new_int( el.GetValue( i ) ));
            }
          }
        break;
      case VR::US:
          {
          Element<VR::US,VM::VM1_n> el;
          el.Set( de.GetValue() );
          int ellen = el.GetLength();
          for( int i = 0; i < ellen; ++i )
            {
            json_object_array_add(my_array, json_object_new_int( el.GetValue( i ) ));
            }
          }
        break;
      case VR::SL:
          {
          Element<VR::SL,VM::VM1_n> el;
          el.Set( de.GetValue() );
          int ellen = el.GetLength();
          for( int i = 0; i < ellen; ++i )
            {
            json_object_array_add(my_array, json_object_new_int( el.GetValue( i ) ));
            }
          }
        break;
      case VR::UL:
          {
          Element<VR::UL,VM::VM1_n> el;
          el.Set( de.GetValue() );
          int ellen = el.GetLength();
          for( int i = 0; i < ellen; ++i )
            {
            json_object_array_add(my_array, json_object_new_int( el.GetValue( i ) ));
            }
          }
        break;
      case VR::AT:
          {
          Element<VR::AT,VM::VM1_n> el;
          el.Set( de.GetValue() );
          int ellen = el.GetLength();
          for( int i = 0; i < ellen; ++i )
            {
            const std::string atstr = el.GetValue( i ).PrintAsContinuousUpperCaseString();
            json_object_array_add(my_array, json_object_new_string( atstr.c_str() ));
            }
          }
        break;
      case VR::UN:
      case VR::INVALID:
      case VR::OD:
      case VR::OF:
      case VR::OB:
      case VR::OW:
          {
          assert( !de.IsUndefinedLength() ); // handled before
          const ByteValue * bv = de.GetByteValue();
          wheretostore = "InlineBinary";
          if( bv )
            {
            const char *src = bv->GetPointer();
            const size_t len = bv->GetLength();
            assert( len % 2 == 0 );
            const size_t len64 = Base64::GetEncodeLength(src, len);
            buffer.resize( len64 );
            const size_t ret = Base64::Encode( &buffer[0], len64, src, len );
            assert( ret != 0 );
            json_object_array_add(my_array, json_object_new_string_len(&buffer[0], len64));
            }
          }
        break;
      default:
        assert( 0 ); // programmer error
        }
      json_object_object_add(my_object_cur, wheretostore, my_array );
      }
    //const char *keyword = entry.GetKeyword();
    //assert( keyword && *keyword );
    //if( preferkeyword && keyword && *keyword && !t.IsPrivateCreator() )
    //  {
    //  json_object_object_add(my_object, keyword, my_object_cur );
    //  }
    //else
      {
      json_object_object_add(my_object, str_tag.c_str(), my_object_cur );
      }
    }
}
#endif

bool JSON::Code(DataSet const & ds, std::ostream & os)
{
#ifdef GDCM_USE_SYSTEM_JSON
  json_object *my_object;
  my_object = json_object_new_object();

  ProcessNestedDataSet( ds, my_object, Internals->PreferKeyword );

  const char* str = NULL;
  if( Internals->PrettyPrint )
    {
#ifdef JSON_C_VERSION
    str = json_object_to_json_string_ext(my_object, JSON_C_TO_STRING_SPACED | JSON_C_TO_STRING_PRETTY );
#else
    str = json_object_to_json_string( my_object );
#endif
    }
  else
    {
    str = json_object_to_json_string( my_object );
    }
  os << str;
  json_object_put(my_object); // free memory
  return true;
#else
  (void)ds;
  (void)os;
  return false;
#endif
}

#ifdef GDCM_USE_SYSTEM_JSON
// Paranoid
static inline bool CheckTagKeywordConsistency( const char *name, const Tag & thetag )
{
  // Can be keyword or tag
  assert( name );

  // start with easy one:
  // only test first string character:
  bool istag = *name >= '0' && *name <= '9'; // should be relatively efficient
  if( istag )
    {
    assert( strlen(name) == 8 );
    Tag t;
    t.ReadFromContinuousString( name );
    return t == thetag;
    }
  // else keyword:
  const Global& g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();
  const Dict &d = dicts.GetPublicDict();
  const char * keyword = d.GetKeywordFromTag(thetag);
  if( !keyword )
    {
    gdcmDebugMacro( "Unknown Keyword: " << name );
    return true;
    }
  // else
  assert( strcmp( name, keyword ) == 0 );
  return strcmp( name, keyword ) == 0;
}
#endif

#ifdef GDCM_USE_SYSTEM_JSON
#ifdef JSON_C_VERSION
static void ProcessJSONElement( const char *tag_str, json_object * obj, DataElement & de )
{
  json_type jtype = json_object_get_type( obj );
  assert( jtype == json_type_object );
  json_object * jvr = json_object_object_get(obj, "VR");

  const char * vr_str = json_object_get_string ( jvr );
  de.GetTag().ReadFromContinuousString( tag_str );
  const char * pc_str = 0;
  if( de.GetTag().IsPrivate() && !de.GetTag().IsPrivateCreator() )
    {
    json_object * jprivatecreator = json_object_object_get(obj, "PrivateCreator");
    pc_str = json_object_get_string ( jprivatecreator );
    assert( pc_str );
    }

  VR::VRType vrtype = VR::GetVRTypeFromFile( vr_str );
  assert( vrtype != VR::INVALID );
  assert( vrtype != VR::VR_END );
  de.SetVR( vrtype );

  if( vrtype == VR::SQ )
    {
    json_object * jvalue = json_object_object_get(obj, "Value");
    json_type jvaluetype = json_object_get_type( jvalue );
    assert( jvaluetype != json_type_null && jvaluetype == json_type_array  );
#ifndef NDEBUG
    json_object * jseq = json_object_object_get(obj, "Sequence");
    json_type jsqtype = json_object_get_type( jseq );
    assert( jsqtype == json_type_null );
#endif
    if( jvaluetype == json_type_array )
      {
      // Create a Sequence
      SmartPointer<SequenceOfItems> sq = new SequenceOfItems;
      sq->SetLengthToUndefined();

      int sqlen = json_object_array_length ( jvalue );
      for( int itemidx = 0; itemidx < sqlen; ++itemidx )
        {
        json_object * jitem = json_object_array_get_idx ( jvalue, itemidx );
        json_type jitemtype = json_object_get_type( jitem );
        assert( jitemtype == json_type_object );
        //const char * dummy = json_object_to_json_string ( jitem );

        // Create an item
        Item item;
        item.SetVLToUndefined();
        DataSet &nds = item.GetNestedDataSet();

#ifdef JSON_C_VERSION
        json_object_iterator it;
        json_object_iterator itEnd;
        it = json_object_iter_begin(jitem);
        itEnd = json_object_iter_end(jitem);

        while (!json_object_iter_equal(&it, &itEnd))
          {
          const char *name = json_object_iter_peek_name(&it);
          assert( name );
          json_object * value = json_object_iter_peek_value (&it);
          DataElement lde;
          ProcessJSONElement( name, value, lde );
          nds.Insert( lde );
          json_object_iter_next(&it);
          }
#endif
        sq->AddItem(item);
        }

      // Insert sequence into data set
      de.SetValue(*sq);
      de.SetVLToUndefined();
      }
    }
  else if( VR::IsASCII( vrtype ) )
    {
/*
    F.2.5              DICOM JSON Model Null Values
    If an attribute is present in DICOM but empty, it shall be preserved in the DICOM JSON object and passed
    with the value of "null". For example:
    "Value": [ null ]
*/
    json_object * jvalue = json_object_object_get(obj, "Value");
#ifndef NDEBUG
    json_object * jpn = json_object_object_get(obj, "PersonName");
    json_type jpntype = json_object_get_type( jpn );
    assert( jpntype == json_type_null );
#endif
    json_type jvaluetype = json_object_get_type( jvalue );
    //const char * dummy = json_object_to_json_string ( jvalue );
    assert( jvaluetype == json_type_null || jvaluetype == json_type_array );
    if( jvaluetype == json_type_array )
      {
      //assert( vrtype != VR::PN );
      const int valuelen = json_object_array_length ( jvalue );
      std::string str;
      for( int validx = 0; validx < valuelen; ++validx )
        {
        if( validx ) str += '\\';
        json_object * value = json_object_array_get_idx ( jvalue, validx );
        json_type valuetype = json_object_get_type( value );
        if( value )
          {
          assert( valuetype != json_type_null );
          std::string value_str;
          std::stringstream ss;
          VRToType<VR::IS>::Type vris;
          VRToType<VR::DS>::Type vrds;
          switch( vrtype )
            {
          case VR::PN:
              {
              json_object * jopn[3];
              jopn[0] = json_object_object_get(value, "Alphabetic");
              jopn[1]= json_object_object_get(value, "Ideographic");
              jopn[2]= json_object_object_get(value, "Phonetic");
              for( int i = 0; i < 3; ++i )
                {
                const char *tmp = json_object_get_string ( jopn[i] );
                if( tmp )
                  {
                  if( i ) value_str += '=';
                  value_str += tmp;
                  }
                }
              }
            break;
          case VR::IS:
            vris = json_object_get_int( value );
            ss << vris;
            value_str = ss.str();
            break;
          case VR::DS:
            vrds = json_object_get_double( value );
            ss << vrds;
            value_str = ss.str();
            break;
          default:
            value_str = json_object_get_string ( value );
            }
          str += value_str;
          }
        else
          {
          // We have a [ null ] array, so at most there is a single item:
          assert( valuelen == 1 );
          assert( valuetype == json_type_null );
          }
        }
      if( str.size() % 2 )
        {
        if( vrtype == VR::UI )
          str.push_back( 0 );
        else
          str.push_back( ' ' );
        }
      de.SetByteValue( &str[0], str.size() );
      }
#ifndef NDEBUG
    else if( jpntype == json_type_array )
      {
      assert( 0 );
      }
#endif
    }
  else
    {
    json_object * jvaluebin = json_object_object_get(obj, "InlineBinary");
    json_type jvaluebintype = json_object_get_type( jvaluebin );
    json_object * jvalue = json_object_object_get(obj, "Value");
    json_type jvaluetype = json_object_get_type( jvalue );
    //const char * dummy = json_object_to_json_string ( jvalue );
    assert( jvaluetype == json_type_array || jvaluetype == json_type_null );
    if( jvaluetype == json_type_array )
      {
      DataElement locde;
      const int valuelen = json_object_array_length ( jvalue );
      const int vrsizeof = vrtype == VR::INVALID ? 0 : de.GetVR().GetSizeof();
      switch( vrtype )
        {
      case VR::FD:
          {
          Element<VR::FD,VM::VM1_n> el;
          el.SetLength( valuelen * vrsizeof );
          for( int validx = 0; validx < valuelen; ++validx )
            {
            json_object * value = json_object_array_get_idx ( jvalue, validx );
            assert( json_object_get_type( value ) == json_type_double );
            const double v = json_object_get_double ( value );
            el.SetValue(v, validx);
            }
          locde = el.GetAsDataElement();
          }
        break;
      case VR::FL:
          {
          Element<VR::FL,VM::VM1_n> el;
          el.SetLength( valuelen * vrsizeof );
          for( int validx = 0; validx < valuelen; ++validx )
            {
            json_object * value = json_object_array_get_idx ( jvalue, validx );
            assert( json_object_get_type( value ) == json_type_double );
            const double v = json_object_get_double ( value );
            el.SetValue(v, validx);
            }
          locde = el.GetAsDataElement();
          }
        break;
      case VR::SS:
          {
          Element<VR::SS,VM::VM1_n> el;
          el.SetLength( valuelen * vrsizeof );
          for( int validx = 0; validx < valuelen; ++validx )
            {
            json_object * value = json_object_array_get_idx ( jvalue, validx );
            assert( json_object_get_type( value ) == json_type_int );
            const int v = json_object_get_int( value );
            el.SetValue(v, validx);
            }
          locde = el.GetAsDataElement();
          }
        break;
      case VR::US:
          {
          Element<VR::US,VM::VM1_n> el;
          el.SetLength( valuelen * vrsizeof );
          for( int validx = 0; validx < valuelen; ++validx )
            {
            json_object * value = json_object_array_get_idx ( jvalue, validx );
            assert( json_object_get_type( value ) == json_type_int );
            const int v = json_object_get_int( value );
            el.SetValue(v, validx);
            }
          locde = el.GetAsDataElement();
          }
        break;
      case VR::SL:
          {
          Element<VR::SL,VM::VM1_n> el;
          el.SetLength( valuelen * vrsizeof );
          for( int validx = 0; validx < valuelen; ++validx )
            {
            json_object * value = json_object_array_get_idx ( jvalue, validx );
            assert( json_object_get_type( value ) == json_type_int );
            const int v = json_object_get_int( value );
            el.SetValue(v, validx);
            }
          locde = el.GetAsDataElement();
          }
        break;
      case VR::UL:
          {
          Element<VR::UL,VM::VM1_n> el;
          el.SetLength( valuelen * vrsizeof );
          for( int validx = 0; validx < valuelen; ++validx )
            {
            json_object * value = json_object_array_get_idx ( jvalue, validx );
            assert( json_object_get_type( value ) == json_type_int );
            const int v = json_object_get_int( value );
            el.SetValue(v, validx);
            }
          locde = el.GetAsDataElement();
          }
        break;
      case VR::AT:
          {
          Element<VR::AT,VM::VM1_n> el;
          el.SetLength( valuelen * vrsizeof );
          for( int validx = 0; validx < valuelen; ++validx )
            {
            json_object * value = json_object_array_get_idx ( jvalue, validx );
            assert( json_object_get_type( value ) == json_type_string );
            const char *atstr = json_object_get_string( value );
            Tag t;
            t.ReadFromContinuousString( atstr );
            el.SetValue(t, validx);
            }
          locde = el.GetAsDataElement();
          }
        break;
      default:
        assert( 0 );
        }
      if( !locde.IsEmpty() )
        de.SetValue( locde.GetValue() );
      }
    else if( jvaluebintype == json_type_array )
      {
      DataElement locde;
      const int valuelen = json_object_array_length ( jvaluebin );
      switch( vrtype )
        {
      case VR::UN:
      case VR::INVALID:
      case VR::OB:
      case VR::OD:
      case VR::OF:
      case VR::OW:
          {
          assert( valuelen == 1 || valuelen == 0 );
          if( valuelen )
            {
            json_object * value = json_object_array_get_idx ( jvaluebin, 0 );
            json_type valuetype = json_object_get_type( value );
            if( value )
              {
              assert( valuetype != json_type_null );
              const char * value_str = json_object_get_string ( value );
              assert( value_str );
              const size_t len64 = strlen( value_str );
              const size_t len = Base64::GetDecodeLength( value_str, len64 );
              std::vector<char> buffer;
              buffer.resize( len );
              const size_t ret = Base64::Decode( &buffer[0], len,
                value_str, len64 );
              assert( ret != 0 );
              locde.SetByteValue( &buffer[0], len );
              }
            else
              {
              // We have a [ null ] array, so at most there is a single item:
              assert( valuelen == 1 );
              assert( valuetype == json_type_null );
              }
            }
          }
        break;
      default:
        assert( 0 );
        }
      if( !locde.IsEmpty() )
        de.SetValue( locde.GetValue() );
      }
    else
      {
      assert( jvaluebintype == json_type_null && jvaluetype == json_type_null );
      }
    }
}
#endif
#endif

bool JSON::Decode(std::istream & is, DataSet & ds)
{
#ifdef GDCM_USE_SYSTEM_JSON

#ifdef JSON_C_VERSION
  json_object *jobj = NULL;
  const char *mystring = NULL;
  int stringlen = 0;
  enum json_tokener_error jerr;
  std::string str;
  json_tokener * tok = json_tokener_new ();
  do
    {
    std::getline( is, str );
    mystring = str.c_str();
    stringlen = str.size();
    jobj = json_tokener_parse_ex(tok, mystring, stringlen);
    //if( is.eof() ) break;
    } while ((jerr = json_tokener_get_error(tok)) == json_tokener_continue );

  if (jerr != json_tokener_success)
    {
    fprintf(stderr, "Error: %s\n", json_tokener_error_desc(jerr));
    // Handle errors, as appropriate for your application.
    assert( 0 );
    }
  if (tok->char_offset < stringlen) // XXX shouldn't access internal fields
    {
    // Handle extra characters after parsed object as desired.
    // e.g. issue an error, parse another object from that point, etc...
    }
  // Success, use jobj here.
  json_tokener_free( tok );
#else
  std::stringstream ss;
  std::string str;
  while( std::getline( is, str ) )
    {
    ss << str;
    }
  const std::string & wholestr = ss.str();
  json_object *obj;
  obj = json_tokener_parse( wholestr.c_str() );
#endif

#ifdef JSON_C_VERSION
  json_object_iterator it;
  json_object_iterator itEnd;
  it = json_object_iter_begin(jobj);
  itEnd = json_object_iter_end(jobj);

  while (!json_object_iter_equal(&it, &itEnd))
    {
    const char *name = json_object_iter_peek_name(&it);
    assert( name );
    json_object * value = json_object_iter_peek_value (&it);
    DataElement de;
    ProcessJSONElement( name, value, de );
    ds.Insert( de );
    json_object_iter_next(&it);
    }
  return true;
#else
  gdcmErrorMacro( "Version too old" );
  return false;
#endif
#else
  (void)is;
  (void)ds;
  return false;
#endif
}

} // end namespace gdcm
