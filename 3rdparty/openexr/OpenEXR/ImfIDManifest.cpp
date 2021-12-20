// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.

//-----------------------------------------------------------------------------
//
//        ID Manifest class implementation
//
//-----------------------------------------------------------------------------

#include <ImfIDManifest.h>
#include <Iex.h>
#include <zlib.h>
#include "ImfXdr.h"
#include "ImfIO.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

//
// debugging only
//
#ifdef DUMP_TABLE
#include <iostream>
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;
using std::make_pair;
using std::set;
using std::vector;
using std::map;
using std::string;
using std::pair;
using std::sort;
using std::fill;

const std::string IDManifest::UNKNOWN = "unknown";
const std::string IDManifest::NOTHASHED = "none";
const std::string IDManifest::CUSTOMHASH = "custom";
const std::string IDManifest::MURMURHASH3_32 = "MurmurHash3_32";
const std::string IDManifest::MURMURHASH3_64 = "MurmurHash3_64";

const std::string IDManifest::ID_SCHEME = "id";
const std::string IDManifest::ID2_SCHEME = "id2";


IDManifest::IDManifest() { }

namespace
{
    
      
// map of strings to index of string in table
    typedef std::map<std::string,int> indexedStringSet;
        
    
    
    // when handling vectors/sets of strings, the string is got by deferencing the pointer/iterator
    template<class T> size_t stringSize(const T &i)
    {
       return i->size();   
    }
    
    template<class T> const char* cStr(const T &i)
    {
       return i->c_str();
    }
    
    /*
    // but for indexedStringSet the string is the first of the iterator pair
    size_t stringSize(indexedStringSet::const_iterator &i )
    {
        return i->first.size();
    }
    
    const char* cStr(indexedStringSet::const_iterator &i)
    {
       return i->first.c_str();
    }
    */

    size_t getVariableLengthIntegerSize(uint64_t value)
    {    

       if(value < 1llu<<7)
       {
           return 1;
       }
       
       if(value < 1llu<<14)
       {
           return 2;
       }      
       if(value < 1llu<<21)
       {
           return 3;
       }
       if(value < 1llu<<28)
       {
           return 4;
       }
       if(value < 1llu<<35)
       {
           return 5;
       }
       if(value < 1llu<<42)
       {
           return 6;
       }
       if(value < 1llu<<49)
       {
           return 7;
       }
       if(value < 1llu<<56)
       {
           return 8;
       }
       if(value < 1llu<<63)
       {
           return 9;
       }
       return 10;
       
    }

    uint64_t readVariableLengthInteger(const char*& readPtr,const char* endPtr)
    {
        // bytes are stored LSB first, so each byte that is read from the stream must be
        // shifted before mixing into the existing length
        int shift=0;
        unsigned char byte=0;
        uint64_t value=0;
        do{
            if(readPtr>=endPtr)
            {
                throw IEX_NAMESPACE::InputExc ("IDManifest too small for variable length integer");
            }
            byte =  *(unsigned char*)readPtr++;
            // top bit of byte isn't part of actual number, it just indicates there's more info to come
            // so take bottom 7 bits, shift them to the right place, and insert them
            //
            value|=(uint64_t(byte&127)) << shift; 
            shift+=7;
        }while(byte&128); //while top bit set on previous byte, there is more to come
        return value;
    }
    
    void writeVariableLengthInteger(char*& outPtr,uint64_t value)
    {
        do
        {
            unsigned char byte = (unsigned char)(value&127);
            value>>=7;
            if(value>0)
            {
                byte|=128;
            }
            *(unsigned char*) outPtr++ = byte;
        }
        while(value>0);
    }
    

    
    //
    // read a list of strings into the given container
    // format is:
    // numberOfStrings (unless numberOfStrings already passed in)
    //  length of string 0
    //  length of string 1
    //  ...
    //  string 0
    //  string 1
    //  ...
    //  (the sizes come first then the strings because that helps compression performance)
    //  note - updates readPtr to point to first byte after readStrings
    //
    
    
    template<class T> void readStringList(const char*& readPtr,const char* endPtr,T & outputVector,int numberOfStrings=0)
    {
        if(numberOfStrings==0)
        {
            if(readPtr+4>endPtr)
            {
                throw IEX_NAMESPACE::InputExc ("IDManifest too small for string list size");
            }
            Xdr::read<CharPtrIO>(readPtr,numberOfStrings);
        }
        
        vector<size_t> lengths(numberOfStrings);
        
        for( int i=0 ; i<numberOfStrings ; ++i )
        {
            lengths[i] = readVariableLengthInteger(readPtr,endPtr);
        }
        for( int i=0 ; i<numberOfStrings ; ++i )
        {

            if(readPtr+lengths[i]  > endPtr)
            {
                throw IEX_NAMESPACE::InputExc ("IDManifest too small for string");
            }
            outputVector.insert(outputVector.end(),string(readPtr , lengths[i]));
            readPtr += lengths[i];
        }
    }
    
    //
    // computes number of bytes required to serialize vector/set of strings
    //
    template<typename T> int getStringListSize(const T& stringList , size_t entries=0)
    {
        int totalSize=0;
        if(entries==0)
        {
            totalSize+=4; // 4 bytes to store number of entries;
        }
        else
        {
            if(stringList.size()!=entries)
            {
                  throw IEX_NAMESPACE::InputExc ("Incorrect number of components stored in ID Manifest");
            }
        }
        for(typename T::const_iterator i = stringList.begin() ; i!=stringList.end() ; ++i)
        {
            size_t length = stringSize(i);
            totalSize+=length;
            // up to five bytes for variable length encoded size
            
            totalSize+=getVariableLengthIntegerSize(length);
            
       
            
        }
        return totalSize;
    }
    
    //
    // write string list to outPtr. if entries nonzero, omits number of entries,
    // but confirms 'entries' == T.size()
    //
    template<typename  T> void
    writeStringList( char*& outPtr,const T& stringList,int entries=0)
    {
        int size = stringList.size();
        if(entries==0)
        {
            Xdr::write<CharPtrIO>(outPtr,size);
        }
        else
        {
            if(size!=entries)
            {
                  throw IEX_NAMESPACE::InputExc ("Incorrect number of components stored in ID Manifest");
            }
        }
        for(typename T::const_iterator i = stringList.begin() ; i!=stringList.end() ; ++i)
        {
            int stringLength = stringSize(i);
            //
            // variable length encoding:
            // values between 0 and 127 inclusive are stored in a single byte
            // values betwwen 128 and 16384 are encoded with two bytes: 1LLLLLLL 0MMMMMMMM where L and M are the least and most significant bits of the value
            // in general, values are stored least significant values first, with the top bit of each byte indicating more values follow
            // the top bit is clear in the last byte of the value
            // (this scheme requires two bytes to store values above 1<<7, and five bytes to store values above 1<<28)
            //
            
            writeVariableLengthInteger(outPtr,stringLength);
            
            
        }
        
        for(typename T::const_iterator i = stringList.begin() ; i!=stringList.end() ; ++i)
        {    
            int stringLength = stringSize(i);
            Xdr::write<CharPtrIO>(outPtr,(const char*) cStr(i),stringLength);
        }
    }
    
    int getStringSize(const string & str)
    {
        return 4+str.size();
    }
    
    void readPascalString(const char*& readPtr, const char* endPtr, string& outputString)
    {

        if(readPtr+4>endPtr)
        {
            throw IEX_NAMESPACE::InputExc ("IDManifest too small for string size");
            
        }
        unsigned int length=0;
        Xdr::read<CharPtrIO>(readPtr,length);

        if(readPtr+length>endPtr)
        {
             throw IEX_NAMESPACE::InputExc ("IDManifest too small for string");
        }
        outputString = string((const char*) readPtr,length);
        readPtr+=length;
    }
    
    void
    writePascalString(char*& outPtr,const string& str)
    {
        unsigned int length = str.size();
        Xdr::write<CharPtrIO>((char*&) outPtr,length);
        Xdr::write<CharPtrIO>((char*&) outPtr,(const char*) str.c_str(),length);
    }


    
}

IDManifest::IDManifest(const char* data,const char* endOfData)
{
    init(data,endOfData);
}

void IDManifest::init(const char* data, const char* endOfData)
{
 

 
   unsigned int version;
   Xdr::read<CharPtrIO>( data,version);
   if(version!=0)
   {
       throw IEX_NAMESPACE::InputExc ("Unrecognized IDmanifest version");
   }
       
   //
   // first comes list of all strings used in manifest
   //
   vector<string> stringList;
   readStringList(data,endOfData,stringList);
   
   //
   // expand the strings in the stringlist
   // each string begins with number of characters to copy from the previous string
   // the remainder is the 'new' bit that appears after that
   //

   for(size_t i=1;i<stringList.size();++i)
   {
      
      size_t common; // number of characters in common with previous string
      int stringStart=1; // first character of string itself;
      //
      // previous string had more than 255 characters? 
      //
      if(stringList[i-1].size()>255)
      {
          common = size_t( ((unsigned char) (stringList[i][0])) <<8 ) + size_t( (unsigned char) (stringList[i][1]));
          stringStart=2;
      }
      else
      {
          common = (unsigned char) stringList[i][0];
      }
      if(common>stringList[i-1].size())
      {
          throw IEX_NAMESPACE::InputExc ("Bad common string length in IDmanifest string table");
      }
      stringList[i] = stringList[i-1].substr(0,common)+stringList[i].substr(stringStart);
          
   }
   
   //
   // decode mapping table from indices in table to indices in string list
   // the mapping uses smaller indices for more commonly occuring strings, since these are encoded with fewer bits
   // comments in serialize function describe the format
   //
   
   vector<int> mapping(stringList.size());
   
   
   // 
   // overlapping sequences: A list [(4,5),(3,6)] expands to 4,5,3,6 - because 4 and 5 are including already
   // they are not included again
   // the 'seen' list indicates which values have already been used, so they are not re-referenced
   //

   vector<char> seen(stringList.size());
   
   int rleLength;
   if(endOfData<data+4)
   {
       throw IEX_NAMESPACE::InputExc ("IDManifest too small");
   }
   
   Xdr::read<CharPtrIO>( data,rleLength);
   
   int currentIndex=0;
   for(int i=0;i<rleLength;++i)
   {
       int first;
       int last;
       if(endOfData<data+8)
       {
           throw IEX_NAMESPACE::InputExc ("IDManifest too small");
       }
       Xdr::read<CharPtrIO>( data,first);
       Xdr::read<CharPtrIO>( data,last);
       
       if(first<0 || last<0 || first>last || first>= int(stringList.size()) || last>=int(stringList.size()))
       {
           throw IEX_NAMESPACE::InputExc ("Bad mapping table entry in IDManifest");
       }
       for(int entry=first ; entry<=last;entry++)
       {
           // don't remap already mapped values
           if(seen[entry]==0)
           {
               mapping[currentIndex]=entry;
               seen[entry]=1;
               currentIndex++;
           }
       }
       
   }
   
#ifdef DUMP_TABLE   
   //
   // dump mapping table for debugging
   //
   for(size_t i=0;i<mapping.size();++i)
   {
       std::cout << i << ' ' << mapping[i] << std::endl;
   }
#endif
   
   
   
   
   //
   // number of manifest entries comes after string list
   //
   int manifestEntries;
      
   if(endOfData<data+4)
   {
       throw IEX_NAMESPACE::InputExc ("IDManifest too small");
   }
   
   Xdr::read<CharPtrIO>( data,manifestEntries);
   
   _manifest.clear();

   _manifest.resize(manifestEntries);
   
   for(int manifestEntry = 0 ; manifestEntry < manifestEntries ; ++manifestEntry)
   {

       ChannelGroupManifest& m = _manifest[manifestEntry];
       
       //
       // read header of this manifest entry
       //
       readStringList(data,endOfData,m._channels);
       readStringList(data,endOfData,m._components);

       char lifetime;
       if(endOfData<data+4)
       {
           throw IEX_NAMESPACE::InputExc ("IDManifest too small");
       }
       Xdr::read<CharPtrIO>(data,lifetime);
       


       m.setLifetime(IdLifetime(lifetime));
       readPascalString(data,endOfData,m._hashScheme);
       readPascalString(data,endOfData,m._encodingScheme);

       if(endOfData<data+5)
       {
           throw IEX_NAMESPACE::InputExc ("IDManifest too small");
       }
       char storageScheme;
       Xdr::read<CharPtrIO>(data,storageScheme);
       
       int tableSize;
       Xdr::read<CharPtrIO>(data,tableSize);
       
       uint64_t previousId=0;
       
       for(int entry = 0 ; entry < tableSize ; ++entry)
       {
           uint64_t id;
      
           switch(storageScheme)
           {
               case 0 : 
               {
                   if(endOfData<data+8)
                   {
                       throw IEX_NAMESPACE::InputExc ("IDManifest too small");   
                   }
                   Xdr::read<CharPtrIO>(data,id);
                   break;
               }
               case 1 :
               {
                   if(endOfData<data+4)
                   {
                       throw IEX_NAMESPACE::InputExc ("IDManifest too small");  
                   }
                   unsigned int id32;
                   Xdr::read<CharPtrIO>(data,id32);
                   id = id32;
                   break;
               }
               default :
               {
                   id = readVariableLengthInteger(data,endOfData);
               }
               
           }
           
           id+=previousId;
           previousId=id;
           
           //
           // insert into table - insert tells us if it was already there
           //
           pair< map< uint64_t, vector<string> >::iterator,bool> insertion = m._table.insert(make_pair(id,vector<string>()));
           if(insertion.second==false)
           {
               throw IEX_NAMESPACE::InputExc("ID manifest contains multiple entries for the same ID");
           }
           (insertion.first)->second.resize(m.getComponents().size());
           for(size_t i=0;i<m.getComponents().size();++i)
           {
               int stringIndex = readVariableLengthInteger(data,endOfData);
               if(size_t(stringIndex)>stringList.size() || stringIndex<0)
               {
                   throw IEX_NAMESPACE::InputExc ("Bad string index in IDManifest");
               }
               (insertion.first)->second[i]=stringList[ mapping[ stringIndex] ];
           }
       }
   }
}


IDManifest::IDManifest(const CompressedIDManifest& compressed)
{
   //
   // decompress the compressed manifest
   //
    
    
   vector<Bytef> uncomp(compressed._uncompressedDataSize);
   uLongf outSize = compressed._uncompressedDataSize;
   if(Z_OK != ::uncompress(&uncomp[0] , &outSize , (const Bytef*) compressed._data, compressed._compressedDataSize))
   {
       throw IEX_NAMESPACE::InputExc ("IDManifest decompression (zlib) failed.");
   }
   if(outSize!=compressed._uncompressedDataSize)
   {
       throw IEX_NAMESPACE::InputExc ("IDManifest decompression (zlib) failed: mismatch in decompressed data size");
   }
   
   init((const char*) &uncomp[0],(const char*) &uncomp[0] + outSize);
  
   
   
}


void IDManifest::serialize(std::vector< char >& data) const
{
    
  
   
   indexedStringSet stringSet;

   //
   // build string map - this turns unique strings into indices
   // the manifest stores the string indices - this allows duplicated
   // strings to point to the same place
   // grabs all the strings regardless of which manifest/mapping they are in
   //
   // at this point we just count the manifest entries
   //
   {
        //
        // over each channel group
        //
       for(size_t m=0;m<_manifest.size();++m)
       {       
           // over each mapping
           for( IDManifest::ChannelGroupManifest::IDTable::const_iterator i = _manifest[m]._table.begin(); i!=_manifest[m]._table.end(); ++i)
           {
               // over each string in the mapping
               
               for(size_t s = 0 ; s < i->second.size() ; ++s)
               {
                   stringSet[ i->second[s] ]++;
               }
           }
       }
   }
   

   
   //
   // build compressed string representation - all but first string starts with number of characters to copy from previous string.
   // max 65535 bytes - use two bytes to store if previous string was more than 255 characters, big endian
   //
   vector<string> prefixedStringList(stringSet.size());
   
   //
   // also make a sorted list so the most common entry appears first. Keep equally likely entries in numerical order
   //
   vector< pair<int,int> > sortedIndices(stringSet.size());
   
   string prevString;
   int index=0;
   for( indexedStringSet::iterator i = stringSet.begin() ; i!= stringSet.end() ; ++i)
   {
   
       
       // no prefix on first string - map stores index of each string, so use that rather than a counter;
       if(index==0)
       {
           prefixedStringList[index]=i->first;
       }
       else
       {
           size_t common=0;
           while(common<65535 
               && common < prevString.size()
               && common < i->first.size()
               && prevString[common] == i->first[common])
           {
               ++common;
           }
           
           if(prevString.size()>255)
           {
               //
               // long previous string - use two bytes to encode number of common chars
               //
               prefixedStringList[index] = string( 1, char(common>>8)) + string(1,char(common&255)) + i->first.substr(common);
           }
           else
           {
               prefixedStringList[index] = string( 1,char(common)) + i->first.substr(common);
           }
           
       }
       prevString = i->first;
       sortedIndices[index].first = -i->second; // use negative of count so largest count appears first
       sortedIndices[index].second = index;
       
       //
       // also, repurpose stringSet so that it maps from string names to indices in the string table
       //
       i->second = index;
       
       index++;
   }
    
   sort( sortedIndices.begin(),sortedIndices.end());
   
   //
   // the first 1<<7 characters will all be encoded with 1 byte, regardless of how common they are
   // the next 1<<14 characters will be encoded with 2 bytes
   // (a full huffman encode would do this at the bit level, not the byte level)
   //
   // the mapping table can be reduced in size by rewriting the IDs to exploit that
   // can rearrange the IDs to have more long runs by sorting numbers
   // that will need the same number of bytes to encode together
   //
   {
       size_t i=0;
   
       for(;i<sortedIndices.size() && i <1<<7;++i)
       {
           sortedIndices[i].first=1;
       }
       for(;i<sortedIndices.size() && i <1<<14;++i)
       {
           sortedIndices[i].first=2;
       }
       for(;i<sortedIndices.size() && i <1<<21;++i)
       {
           sortedIndices[i].first=3;
       }
       for(;i<sortedIndices.size() && i <1<<28;++i)
       {
           sortedIndices[i].first=4;
       }
       for(;i<sortedIndices.size();++i)
       {
           sortedIndices[i].first=5;
       }
   }
   sort( sortedIndices.begin(),sortedIndices.end());
   
   
   
   vector<int> stringIndices(sortedIndices.size());
   
   //
   // table will be stored with RLE encoding - store pairs of 'start index,end index'
   // so, the sequence 10,11,12,1,2,3,4  is stored as [ (10,12) , (1,4)]
   //
   // sequential IDs ignore already referenced IDs, so the sequence  11,9,10,12,13 can be stored as [ (11,11) , (9,13)]
   // on reading, don't reference an entry that has already been seen
   // on writing, need to track which entries have already been stored to allow this overlapping to occur
   //
   
   vector< pair<int,int> > RLEmapping;
   
   if(sortedIndices.size()>0)
   {
       RLEmapping.resize(1);
       RLEmapping[0].first = sortedIndices[0].second;
       RLEmapping[0].second = sortedIndices[0].second;
       
       fill(stringIndices.begin(),stringIndices.end(),-1);
       
       stringIndices[sortedIndices[0].second]=0;
       
       //
       // as the loop below runs, nextToInclude tracks the value that can be merged with the current run length
       // (RLWmapping.back()) - generally this is on more than the current length, but it jumps forward
       // over values already seen
       //
       int nextToInclude=stringIndices[sortedIndices[0].second]+1;
       
       
       for( size_t i=1;i<sortedIndices.size() ; ++i)
       {
           if( sortedIndices[i].second == nextToInclude)
           {
               //
               // this index can be treated as part of the current run, so extend the run to include it
               //
               RLEmapping.back().second=sortedIndices[i].second;
              
           }
           else
           {
               pair<int,int> newEntry(sortedIndices[i].second,sortedIndices[i].second);
               RLEmapping.push_back(newEntry);
           }
           // build mapping for this entry
           stringIndices[sortedIndices[i].second]=i;
           
           // what would the next entry have to be to be included in this run
           // skip over already mapped strings
           nextToInclude=sortedIndices[i].second+1;
               
           
           while ( nextToInclude < int(stringIndices.size()) && stringIndices[nextToInclude]>=0)
           {
               nextToInclude++;
           }
       }
   }
#ifdef DUMP_TABLE
   // dump RLE table for debugging
   for( size_t i=1;i<sortedIndices.size() ; ++i)
   {
       std::cout << i << ' ' << sortedIndices[i].second << std::endl;
   }
#endif
   
   
    
    // now compute size of uncompressed memory block for serialization
    
   int outputSize = 8; // at least need four bytes for integer to store number of channel manifests, plus four bytes to indicate version pattern
   
    outputSize += getStringListSize(prefixedStringList);
    
    //
    // RLE mapping table size - number of entries followed by eight bytes for each run length
    //
    outputSize+= RLEmapping.size()*8+4;
  
    // 
    // track which storage scheme is optimal for storing the IDs of each type
    // ID storage scheme: 0 = 8 bytes per ID, 1 = 4 bytes per ID, 2 = variable
    // 

    std::vector<char> storageSchemes;
    
   for(size_t groupNumber = 0 ; groupNumber < _manifest.size() ; ++groupNumber)
   {
       const ChannelGroupManifest& m = _manifest[groupNumber];
       outputSize += getStringListSize(m._channels); //size of channel group
       outputSize += getStringListSize(m._components); //size of component list
       outputSize += 1; //size of lifetime enum
       outputSize += getStringSize(m._hashScheme);
       outputSize += getStringSize(m._encodingScheme);
       
       outputSize += 1; // ID scheme
       outputSize += 4; // size of storage for number of 32 bit entries in ID table
       
       uint64_t previousId=0;       
       uint64_t IdStorageForVariableScheme = 0;
       bool canUse32Bits = true;
       for( IDManifest::ChannelGroupManifest::IDTable::const_iterator i = m._table.begin(); i!=m._table.end(); ++i)
       {
       
           uint64_t idToStore = i->first-previousId;
           IdStorageForVariableScheme+=getVariableLengthIntegerSize(idToStore);
           if(idToStore >= 1llu<<32)
           {
               canUse32Bits = false;
           }
           previousId=i->first;

           for(size_t s=0;s<m._components.size();++s)
           {
               int stringID = stringSet[ i->second[s] ];
               int idToWrite = stringIndices[stringID];
               outputSize+= getVariableLengthIntegerSize(idToWrite);
           }
       }
       // pick best scheme to use to store IDs
       if(canUse32Bits)
       {
           if(IdStorageForVariableScheme < m._table.size()*4)
           {
               //
               // variable storage smaller than fixed 32 bit, so use that
               //
               storageSchemes.push_back(2);
               outputSize+=IdStorageForVariableScheme;
           }
           else
           {
               //
               // variable scheme bigger than fixed 32 bit, but all ID differences fit into 32 bits
               //
               storageSchemes.push_back(1);
               outputSize+=m._table.size()*4;
           }
       }
       else
       {
           if(IdStorageForVariableScheme < m._table.size()*8)
           {
               //
               // variable storage smaller than fixed 64 bit, so use that
               //
               storageSchemes.push_back(2);
               outputSize+=IdStorageForVariableScheme;
           }
           else
           {
               //
               // variable scheme bigger than fixed 64 bit, and some ID differences bigger than 32 bit
               //
               storageSchemes.push_back(0);
               outputSize+=m._table.size()*8;
           }
       }
       
   }
   
   //
   // resize output array
   //
   data.resize(outputSize);
   
   
   
   //
   // populate output array
   //
   char* outPtr = &data[0];
   
   //
   // zeroes to indicate this is version 0 of the header
   //
   Xdr::write<CharPtrIO>(outPtr, int(0));
   
   //
   // table of strings
   //
    writeStringList(outPtr,prefixedStringList);
    
    
   //
   // RLE block
   //
   Xdr::write<CharPtrIO>(outPtr, int(RLEmapping.size()));
   for(size_t i=0;i<RLEmapping.size();++i)
   {
       Xdr::write<CharPtrIO>(outPtr, RLEmapping[i].first);
       Xdr::write<CharPtrIO>(outPtr, RLEmapping[i].second);
   }
    
   
    
   //
   // number of manifests
   //
   Xdr::write<CharPtrIO>(outPtr, int(_manifest.size()));
   int manifestIndex=0;
   
  for(size_t groupNumber = 0 ; groupNumber < _manifest.size() ; ++groupNumber)
   {
       const ChannelGroupManifest& m = _manifest[groupNumber]; 
       //
       // manifest header
       // 
       writeStringList(outPtr,m._channels);
       writeStringList(outPtr,m._components);
       Xdr::write<CharPtrIO>(outPtr,char(m._lifeTime));
       writePascalString(outPtr,m._hashScheme);
       writePascalString(outPtr,m._encodingScheme);
       
       char scheme = storageSchemes[manifestIndex];
       Xdr::write<CharPtrIO>(outPtr,scheme);
       
       Xdr::write<CharPtrIO>(outPtr,int(m._table.size()));
       
       uint64_t previousId=0;     
       //
       // table
       //
       for( IDManifest::ChannelGroupManifest::IDTable::const_iterator i = m._table.begin(); i!=m._table.end(); ++i)
       {
        
           uint64_t idToWrite = i->first-previousId;
           switch(scheme)
           {
               case 0 : Xdr::write<CharPtrIO>(outPtr, idToWrite);break;
               case 1 : Xdr::write<CharPtrIO>(outPtr,(unsigned int) idToWrite);break;
               case 2 : writeVariableLengthInteger(outPtr,idToWrite);
           }
           
           previousId=i->first;
           
           for(size_t s=0;s<m._components.size();++s)
           {
               int stringID = stringSet[ i->second[s] ];
               int idToWrite = stringIndices[stringID];
               writeVariableLengthInteger(outPtr , idToWrite );
           }
       }    
       manifestIndex++;
   }
   //
   // check we've written the ID manifest correctly
   //
   if(outPtr!=&data[0]+data.size())
   {
       throw IEX_NAMESPACE::ArgExc("Error - IDManifest size error");
   }
}

bool
IDManifest::operator==(const IDManifest& other) const
{
   return other._manifest == _manifest;
}

bool
IDManifest::operator!=(const IDManifest& other) const
{
   return !(*this==other);
}

bool
IDManifest::merge(const IDManifest& other)
{
   bool conflict = false;
   for(size_t otherManifest = 0 ; otherManifest < other._manifest.size() ; ++ otherManifest)
   {
       bool merged = false;
       for(size_t thisManifest = 0 ; thisManifest < _manifest.size() ; ++ thisManifest)
       {
           if( _manifest[thisManifest]._channels == other._manifest[otherManifest]._channels)
           {
               // found same channels
               
               merged = true;
               
               if(other._manifest[otherManifest]._components !=  _manifest[thisManifest]._components)
               {
                   // cannot merge if components are different
                   conflict = true;
               }
               else
               {

//                    if(other._manifest[otherManifest]._encodingScheme !=  _manifest[thisManifest]._encodingScheme ||
//                        other._manifest[otherManifest]._hashScheme !=  _manifest[thisManifest]._hashScheme || 
//                        other._manifest[otherManifest]._hashScheme !=  _manifest[thisManifest]._hashScheme ||
//                        other._manifest[otherManifest]._lifeTime !=  _manifest[thisManifest]._lifeTime)
//                    {
//                        conflict = true;
//                    }
                   
                   for( IDManifest::ChannelGroupManifest::ConstIterator it = other._manifest[otherManifest].begin() ; it!= other._manifest[otherManifest].end() ; ++it)
                   {
                      IDManifest::ChannelGroupManifest::ConstIterator ours = _manifest[thisManifest].find( it.id());
                      if( ours ==  _manifest[thisManifest].end())
                      {
                          _manifest[thisManifest].insert( it.id() , it.text());
                      }
                      else
                      {
                          if(ours.text() != it.text())
                          {
                              conflict = true;
                          }
                      }
                   }
               }
           }
       }
       
       if(!merged)
       {
           _manifest.push_back(other._manifest[otherManifest]);
       }
       
       
   }
   
   return conflict;
   
}


CompressedIDManifest::CompressedIDManifest() : _compressedDataSize(0) ,  _uncompressedDataSize(0) , _data(NULL) {}


CompressedIDManifest::CompressedIDManifest(const CompressedIDManifest& other) 
   : _compressedDataSize(other._compressedDataSize) ,
     _uncompressedDataSize(other._uncompressedDataSize) , 
     _data((unsigned char*) malloc(other._compressedDataSize) )
{
   memcpy(_data,other._data,_compressedDataSize);
}

CompressedIDManifest&
CompressedIDManifest::operator=(const CompressedIDManifest& other) 
{
    if(this!=&other)
    {
        if(_data)
        {
            free(_data);
        }
        _data = (unsigned char*)  malloc( other._compressedDataSize);
        _compressedDataSize = other._compressedDataSize;
        _uncompressedDataSize = other._uncompressedDataSize;
        memcpy(_data,other._data,_compressedDataSize);
    }
    return *this;
}



CompressedIDManifest::~CompressedIDManifest()
{
    if(_data)
    {        
        free(_data);
    }
    _data = NULL;
   _compressedDataSize=0;
}


 
CompressedIDManifest::CompressedIDManifest(const IDManifest& manifest)
{
   //
   // make a compressed copy of the manifest by serializing the data into contiguous memory,
   // then calling zlib to compress
   //
   
 
    std::vector<char> serial;
    
    manifest.serialize(serial);
    
    uLong outputSize = serial.size();
   
    //
    // allocate a buffer which is guaranteed to be big enough for compression
    //
   uLongf compressedDataSize = compressBound(outputSize);
   _data = (unsigned char*) malloc(compressedDataSize);
   if(Z_OK != ::compress(_data,&compressedDataSize,(Bytef*) &serial[0],outputSize))
   {
       throw IEX_NAMESPACE::InputExc("ID manifest compression failed");
   }
   
   // now call realloc to reallocate the buffer to a smaller size - this might free up memory
   _data = (unsigned char*) realloc(_data,compressedDataSize);
   
   _uncompressedDataSize = outputSize;
   _compressedDataSize = compressedDataSize;
   
}

IDManifest::ChannelGroupManifest::ChannelGroupManifest() : _lifeTime(IDManifest::LIFETIME_STABLE) , _hashScheme(IDManifest::UNKNOWN) , _encodingScheme(IDManifest::UNKNOWN) , _insertingEntry(false)
{

}

const vector< string >& IDManifest::ChannelGroupManifest::getComponents() const
{
   return _components;
}

set< string >& IDManifest::ChannelGroupManifest::getChannels()
{
  return _channels;
}

const set< string >& IDManifest::ChannelGroupManifest::getChannels() const
{
   return _channels;
}

void IDManifest::ChannelGroupManifest::setChannel(const string& channel)
{
   _channels.clear();
   _channels.insert(channel);
}

void IDManifest::ChannelGroupManifest::setChannels(const set< string >& channels)
{
   _channels = channels;
}






//
// set number of components of table
//
void IDManifest::ChannelGroupManifest::setComponents(const std::vector< std::string >& components)
{

    // if there are already entries in the table, cannot change the number of components
    if(_table.size()!=0 && components.size()!=_components.size())
    {
        THROW (IEX_NAMESPACE::ArgExc, "attempt to change number of components in manifest once entries have been added");
    }
    _components = components;
}

void IDManifest::ChannelGroupManifest::setComponent(const std::string & component)
{
    vector<string> components(1);
    components[0] = component;
    setComponents(components);
}

IDManifest::ChannelGroupManifest::ConstIterator
IDManifest::ChannelGroupManifest::begin() const
{
    return IDManifest::ChannelGroupManifest::ConstIterator(_table.begin());
}

IDManifest::ChannelGroupManifest::Iterator
IDManifest::ChannelGroupManifest::begin()
{
  return IDManifest::ChannelGroupManifest::Iterator(_table.begin());
}

IDManifest::ChannelGroupManifest::ConstIterator
IDManifest::ChannelGroupManifest::end() const
{
   return IDManifest::ChannelGroupManifest::ConstIterator(_table.end());
}

IDManifest::ChannelGroupManifest::Iterator IDManifest::ChannelGroupManifest::end()
{
    return IDManifest::ChannelGroupManifest::Iterator(_table.end());
}

IDManifest::ChannelGroupManifest::ConstIterator
IDManifest::ChannelGroupManifest::find(uint64_t idValue) const
{
    return IDManifest::ChannelGroupManifest::ConstIterator(_table.find(idValue));
}

void
IDManifest::ChannelGroupManifest::erase(uint64_t idValue)
{
   _table.erase(idValue);
}
size_t
IDManifest::ChannelGroupManifest::size() const
{
   return _table.size();
}



IDManifest::ChannelGroupManifest::Iterator IDManifest::ChannelGroupManifest::find(uint64_t idValue)
{
    return IDManifest::ChannelGroupManifest::Iterator(_table.find(idValue));
}

std::vector< std::string >& IDManifest::ChannelGroupManifest::operator[](uint64_t idValue)
{
     return _table[idValue];
}



IDManifest::ChannelGroupManifest::Iterator
IDManifest::ChannelGroupManifest::insert(uint64_t idValue, const std::string& text)
{
    if(_components.size()!=1)
    {
        THROW (IEX_NAMESPACE::ArgExc, "Cannot insert single component attribute into manifest with multiple components");
    }
    vector<string> tempVector(1);
    tempVector[0] = text;
    return IDManifest::ChannelGroupManifest::Iterator(_table.insert(make_pair(idValue,tempVector)).first);
}


IDManifest::ChannelGroupManifest::Iterator
IDManifest::ChannelGroupManifest::insert(uint64_t idValue, const std::vector< std::string >& text)
{
   if(_components.size()!=text.size())
   {
       THROW (IEX_NAMESPACE::ArgExc, "mismatch between number of components in manifest and number of components in inserted entry");
   }
    return IDManifest::ChannelGroupManifest::Iterator(_table.insert(make_pair(idValue,text)).first);
}



uint64_t
IDManifest::ChannelGroupManifest::insert(const std::vector< std::string >& text)
{
    uint64_t hash;
    if(_hashScheme == MURMURHASH3_32)
    {
        hash = MurmurHash32(text);
    }
    else if(_hashScheme == MURMURHASH3_64)
    {
        hash = MurmurHash64(text);
    }
    else
    {
         THROW (IEX_NAMESPACE::ArgExc, "Cannot compute hash: unknown hashing scheme");
    }
    insert(hash,text);
    return hash;
    
}

uint64_t
IDManifest::ChannelGroupManifest::insert(const std::string & text)
{
    uint64_t hash;
    if(_hashScheme == MURMURHASH3_32)
    {
        hash = MurmurHash32(text);
    }
    else if(_hashScheme == MURMURHASH3_64)
    {
        hash = MurmurHash64(text);
    }
    else
    {
         THROW (IEX_NAMESPACE::ArgExc, "Cannot compute hash: unknown hashing scheme");
    }
    insert(hash,text);
    return hash;
}



IDManifest::ChannelGroupManifest&
IDManifest::ChannelGroupManifest::operator<<(uint64_t idValue)
{
   if(_insertingEntry)
   {
        THROW (IEX_NAMESPACE::ArgExc,"not enough components inserted into previous entry in ID table before inserting new entry");
   }
   
   _insertionIterator = _table.insert( make_pair(idValue,std::vector<std::string>() )).first;
   
   //
   // flush out previous entry: reinserting an attribute overwrites previous entry
   //
   _insertionIterator->second.resize(0);
   
   
   //
   // curious edge-case: it's possible to have an ID table with no strings, just a list of IDs
   // There's little purpose to this, but it means that this entry is now 'complete'
   //
   if(_components.size()==0)
   {
       _insertingEntry = false;
   }
   else
   {
       _insertingEntry = true;
   }
     return *this;
}

IDManifest::ChannelGroupManifest&
IDManifest::ChannelGroupManifest::operator<<(const std::string& text)
{
    if(!_insertingEntry)
    {
        THROW (IEX_NAMESPACE::ArgExc,"attempt to insert too many strings into entry, or attempt to insert text before ID integer");
    }
    if(_insertionIterator->second.size() >= _components.size())
    {
        THROW (IEX_NAMESPACE::ArgExc,"Internal error: too many strings in component");
    }
    _insertionIterator->second.push_back(text);
    
    //
    // if the last component has been inserted, switch off insertingEntry, to mark all entries as complete
    //
    if(_insertionIterator->second.size() == _components.size())
    {
        _insertingEntry = false;
    }
    return *this;
}


bool
IDManifest::ChannelGroupManifest::operator==(const IDManifest::ChannelGroupManifest& other) const
{
    return (_lifeTime == other._lifeTime && 
           _components == other._components &&
           _hashScheme == other._hashScheme &&
           _components == other._components &&
           _table == other._table);
           
}


size_t
IDManifest::size() const
{
  return _manifest.size();
}

size_t
IDManifest::find(const string& channel) const
{
    // search the set of channels for each ChannelGroupManifest searching for 
    // one that contains 'channel'
    for( size_t i = 0 ; i < _manifest.size() ; ++i )
    {

        if( _manifest[i].getChannels().find(channel) != _manifest[i].getChannels().end())
        {
            return i;
        }
    }
    //  not find, return size()
    return _manifest.size();
    
}

IDManifest::ChannelGroupManifest&
IDManifest::add(const set< string >& group)
{
   _manifest.push_back(ChannelGroupManifest());
   ChannelGroupManifest& mfst = _manifest.back();
   mfst._channels = group;
   return mfst;
}

IDManifest::ChannelGroupManifest&
IDManifest::add(const string& channel)
{
   _manifest.push_back(ChannelGroupManifest());
   ChannelGroupManifest& mfst = _manifest.back();
   mfst._channels.insert(channel);
   return mfst;
}

IDManifest::ChannelGroupManifest&
IDManifest::add(const IDManifest::ChannelGroupManifest& table)
{
   _manifest.push_back(table);
   return _manifest.back();
}

IDManifest::ChannelGroupManifest& IDManifest::operator[](size_t index)
{
   return _manifest[index];
}

const IDManifest::ChannelGroupManifest& IDManifest::operator[](size_t index) const
{
  return _manifest[index];
}





namespace
{
    
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// 
// smhasher provides two different 128 bit hash schemes, optimised for either
// 32 or 64 bit architectures. IDManifest uses only the 64 bit optimised version
// of the 128 bit hash function to generate '64 bit hashes' 
//-----------------------------------------------------------------------------
// Platform-specific functions and macros
// Microsoft Visual Studio
#if defined(_MSC_VER)
#define FORCE_INLINE          __forceinline
#define ROTL32(x,y) _rotl(x,y)
#define ROTL64(x,y) _rotl64(x,y)
#define BIG_CONSTANT(x) (x)
// Other compilers
#else     // defined(_MSC_VER)
#define   FORCE_INLINE inline __attribute__((always_inline))
inline uint32_t rotl32 ( uint32_t x, int8_t r )
{
  return (x << r) | (x >> (32 - r));
}
inline uint64_t rotl64 ( uint64_t x, int8_t r )
{
  return (x << r) | (x >> (64 - r));
}
#define   ROTL32(x,y)         rotl32(x,y)
#define ROTL64(x,y) rotl64(x,y)
#define BIG_CONSTANT(x) (x##LLU)
#endif // !defined(_MSC_VER)
//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here
FORCE_INLINE uint32_t getblock32 ( const uint32_t * p, int i )
{
  return p[i];
  
}
FORCE_INLINE uint64_t getblock64 ( const uint64_t * p, int i )
{
  return p[i];
}
//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche
FORCE_INLINE uint32_t fmix32 ( uint32_t h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}
//----------
FORCE_INLINE uint64_t fmix64 ( uint64_t k )
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}
//-----------------------------------------------------------------------------
void MurmurHash3_x86_32 ( const void * key, int len,
                          uint32_t seed, void * out )
{
  const uint8_t * data = (const uint8_t*)key;
  const int nblocks = len / 4;
  uint32_t h1 = seed;
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;
  //----------
  // body
  const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);
  for(int i = -nblocks; i; i++)
  {
    uint32_t k1 = getblock32(blocks,i);
    k1 *= c1;
    k1 = ROTL32(k1,15);
    k1 *= c2;
    
    h1 ^= k1;
    h1 = ROTL32(h1,13); 
    h1 = h1*5+0xe6546b64;
  }
  //----------
  // tail
  const uint8_t * tail = (const uint8_t*)(data + nblocks*4);
  uint32_t k1 = 0;
  switch(len & 3)
  {
  case 3: k1 ^= tail[2] << 16;
  case 2: k1 ^= tail[1] << 8;
  case 1: k1 ^= tail[0];
          k1 *= c1; k1 = ROTL32(k1,15); k1 *= c2; h1 ^= k1;
  };
  //----------
  // finalization
  h1 ^= len;
  h1 = fmix32(h1);
  *(uint32_t*)out = h1;
} 

//-----------------------------------------------------------------------------
void MurmurHash3_x64_128 ( const void * key, const int len,
                           const uint32_t seed, void * out )
{
  const uint8_t * data = (const uint8_t*)key;
  const int nblocks = len / 16;
  uint64_t h1 = seed;
  uint64_t h2 = seed;
  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);
  //----------
  // body
  const uint64_t * blocks = (const uint64_t *)(data);
  for(int i = 0; i < nblocks; i++)
  {
    uint64_t k1 = getblock64(blocks,i*2+0);
    uint64_t k2 = getblock64(blocks,i*2+1);
    k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
    h1 = ROTL64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;
    k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;
    h2 = ROTL64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
  }
  //----------
  // tail
  const uint8_t * tail = (const uint8_t*)(data + nblocks*16);
  uint64_t k1 = 0;
  uint64_t k2 = 0;
  switch(len & 15)
  {
      case 15: k2 ^= ((uint64_t)tail[14]) << 48;
      case 14: k2 ^= ((uint64_t)tail[13]) << 40;
      case 13: k2 ^= ((uint64_t)tail[12]) << 32;
      case 12: k2 ^= ((uint64_t)tail[11]) << 24;
      case 11: k2 ^= ((uint64_t)tail[10]) << 16;
      case 10: k2 ^= ((uint64_t)tail[ 9]) << 8;
      case  9: k2 ^= ((uint64_t)tail[ 8]) << 0;
               k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;
      case  8: k1 ^= ((uint64_t)tail[ 7]) << 56;
      case  7: k1 ^= ((uint64_t)tail[ 6]) << 48;
      case  6: k1 ^= ((uint64_t)tail[ 5]) << 40;
      case  5: k1 ^= ((uint64_t)tail[ 4]) << 32;
      case  4: k1 ^= ((uint64_t)tail[ 3]) << 24;
      case  3: k1 ^= ((uint64_t)tail[ 2]) << 16;
      case  2: k1 ^= ((uint64_t)tail[ 1]) << 8;
      case  1: k1 ^= ((uint64_t)tail[ 0]) << 0;
               k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
  };
  //----------
  // finalization
  h1 ^= len; h2 ^= len;
  h1 += h2;
  h2 += h1;
  h1 = fmix64(h1);
  h2 = fmix64(h2);
  h1 += h2;
  h2 += h1;
  ((uint64_t*)out)[0] = h1;
  ((uint64_t*)out)[1] = h2;
}
//-----------------------------------------------------------------------------

//
// combine the idStrings into a single string, separating each with a ; character
// (use of the ; character is discouraged, though not prohibited)
//
void catString(const vector< string >& idString, std::string & str)
{
    str =idString[0];
    for(size_t i = 1 ; i < idString.size() ; ++i)
    {
        str+=";";
        str+=idString[i];
    }
    
}
}

unsigned int IDManifest::MurmurHash32(const std::string & idString)
{
    unsigned int out;
    MurmurHash3_x86_32(idString.c_str(),idString.size(),0 , (void*) &out);
    return out;
}

uint64_t IDManifest::MurmurHash64(const std::string& idString)
{

    uint64_t out[2];
    MurmurHash3_x64_128(idString.c_str(),idString.size(),0 , out);
    return out[0];
}

unsigned int IDManifest::MurmurHash32(const vector< string >& idString)
{
    if(idString.size()==0)
    {
        return 0;
    }
    std::string str;
    catString(idString,str);
    return MurmurHash32(str);
}
 

 
 
    
uint64_t IDManifest::MurmurHash64(const vector< string >& idString)
{
    if(idString.size()==0)
    {
        return 0;
    }
    std::string str;
    catString(idString,str);
    return MurmurHash64(str);
}



OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
