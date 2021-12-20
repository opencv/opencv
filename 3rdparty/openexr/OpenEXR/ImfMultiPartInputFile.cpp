//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfMultiPartInputFile.h"

#include "ImfTimeCodeAttribute.h"
#include "ImfChromaticitiesAttribute.h"
#include "ImfBoxAttribute.h"
#include "ImfFloatAttribute.h"
#include "ImfStdIO.h"
#include "ImfTileOffsets.h"
#include "ImfMisc.h"
#include "ImfTiledMisc.h"
#include "ImfInputStreamMutex.h"
#include "ImfInputPartData.h"
#include "ImfPartType.h"
#include "ImfInputFile.h"
#include "ImfScanLineInputFile.h"
#include "ImfTiledInputFile.h"
#include "ImfDeepScanLineInputFile.h"
#include "ImfDeepTiledInputFile.h"
#include "ImfVersion.h"

#include <OpenEXRConfig.h>

#include <Iex.h>
#include <map>
#include <set>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;

using std::vector;
using std::map;
using std::set;
using std::string;

namespace
{
    // Controls whether we error out in the event of shared attribute
    // inconsistency in the input file
    static const bool strictSharedAttribute = true;
}

struct MultiPartInputFile::Data: public InputStreamMutex
{
    int                         version;        // Version of this file.
    bool                        deleteStream;   // If we should delete the stream during destruction.
    vector<InputPartData*>      parts;          // Data to initialize Output files.
    int                         numThreads;     // Number of threads
    bool                        reconstructChunkOffsetTable;    // If we should reconstruct
                                                                // the offset table if it's broken.
    std::map<int,GenericInputFile*> _inputFiles;
    std::vector<Header>             _headers;

    
    void                    chunkOffsetReconstruction(OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is, const std::vector<InputPartData*>& parts);
                                                      
    void                    readChunkOffsetTables(bool reconstructChunkOffsetTable);
                                                      
    bool                    checkSharedAttributesValues(const Header & src,
                                                        const Header & dst,
                                                        std::vector<std::string> & conflictingAttributes) const;
                                                                                                          
   TileOffsets*            createTileOffsets(const Header& header);
   
   InputPartData*          getPart(int partNumber);
   
    Data (bool deleteStream, int numThreads, bool reconstructChunkOffsetTable):
        InputStreamMutex(),
        deleteStream (deleteStream),
        numThreads (numThreads),
        reconstructChunkOffsetTable(reconstructChunkOffsetTable)
    {
    }

    ~Data()
    {
        if (deleteStream) delete is;

        for (size_t i = 0; i < parts.size(); i++)
            delete parts[i];
    }
    
    Data (const Data& other) = delete;
    Data& operator = (const Data& other) = delete;
    Data (Data&& other) = delete;
    Data& operator = (Data&& other) = delete;
    
    template <class T>
    T*    createInputPartT(int partNumber)
    {

    }
};

MultiPartInputFile::MultiPartInputFile(const char fileName[],
                           int numThreads,
                           bool reconstructChunkOffsetTable):
    _data(new Data(true, numThreads, reconstructChunkOffsetTable))
{
    try
    {
        _data->is = new StdIFStream (fileName);
        initialize();
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _data;

        REPLACE_EXC (e, "Cannot read image file "
                     "\"" << fileName << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        delete _data;
        throw;
    }
}

MultiPartInputFile::MultiPartInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is,
                                        int numThreads,
                                        bool reconstructChunkOffsetTable):
    _data(new Data(false, numThreads, reconstructChunkOffsetTable))
{
    try
    {
        _data->is = &is;
        initialize();
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _data;

        REPLACE_EXC (e, "Cannot read image file "
                     "\"" << is.fileName() << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        delete _data;
        throw;
    }
}

template<class T>
T*
MultiPartInputFile::getInputPart(int partNumber)
{
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock(*_data);
#endif
    if (_data->_inputFiles.find(partNumber) == _data->_inputFiles.end())
    {
        T* file = new T(_data->getPart(partNumber));
        _data->_inputFiles.insert(std::make_pair(partNumber, (GenericInputFile*) file));
        return file;
    }

    return (T*) _data->_inputFiles[partNumber];
}

void
MultiPartInputFile::flushPartCache()
{
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock(*_data);
#endif
    while ( _data->_inputFiles.begin() != _data->_inputFiles.end())
    {
       delete _data->_inputFiles.begin()->second;
       _data->_inputFiles.erase(_data->_inputFiles.begin());
    }
}


template InputFile* MultiPartInputFile::getInputPart<InputFile>(int);
template TiledInputFile* MultiPartInputFile::getInputPart<TiledInputFile>(int);
template DeepScanLineInputFile* MultiPartInputFile::getInputPart<DeepScanLineInputFile>(int);
template DeepTiledInputFile* MultiPartInputFile::getInputPart<DeepTiledInputFile>(int);

InputPartData*
MultiPartInputFile::getPart(int partNumber)
{
    return _data->getPart(partNumber);
}



const Header &
 MultiPartInputFile::header(int n) const
{
    if(n<0 || n >= _data->_headers.size())
    {
        THROW ( IEX_NAMESPACE::ArgExc , " MultiPartInputFile::header called with invalid part " << n << " on file with " << _data->_headers.size() << " parts");
    }
    return _data->_headers[n];
}



MultiPartInputFile::~MultiPartInputFile()
{
    for (map<int, GenericInputFile*>::iterator it = _data->_inputFiles.begin();
         it != _data->_inputFiles.end(); it++)
    {
        delete it->second;
    }

    delete _data;
}


bool
MultiPartInputFile::Data::checkSharedAttributesValues(const Header & src,
                                                const Header & dst,
                                                vector<string> & conflictingAttributes) const
{
    conflictingAttributes.clear();

    bool conflict = false;

    //
    // Display Window
    //
    if (src.displayWindow() != dst.displayWindow())
    {
        conflict = true;
        conflictingAttributes.push_back ("displayWindow");
    }


    //
    // Pixel Aspect Ratio
    //
    if (src.pixelAspectRatio() != dst.pixelAspectRatio())
    {
        conflict = true;
        conflictingAttributes.push_back ("pixelAspectRatio");
    }


    //
    // Timecode
    //
    const TimeCodeAttribute * srcTimeCode = src.findTypedAttribute<
          TimeCodeAttribute> (TimeCodeAttribute::staticTypeName());
    const TimeCodeAttribute * dstTimeCode = dst.findTypedAttribute<
          TimeCodeAttribute> (TimeCodeAttribute::staticTypeName());

    if (dstTimeCode)
    {
        if  ( (srcTimeCode && (srcTimeCode->value() != dstTimeCode->value())) ||
              (!srcTimeCode))
        {
            conflict = true;
            conflictingAttributes.push_back (TimeCodeAttribute::staticTypeName());
        }
    }

    //
    // Chromaticities
    //
    const ChromaticitiesAttribute * srcChrom =  src.findTypedAttribute<
          ChromaticitiesAttribute> (ChromaticitiesAttribute::staticTypeName());
    const ChromaticitiesAttribute * dstChrom =  dst.findTypedAttribute<
          ChromaticitiesAttribute> (ChromaticitiesAttribute::staticTypeName());

    if (dstChrom)
    {
        if ( (srcChrom && (srcChrom->value() != dstChrom->value())) ||
             (!srcChrom))
        {
            conflict = true;
            conflictingAttributes.push_back (ChromaticitiesAttribute::staticTypeName());
        }
    }


    return conflict;
}


void
MultiPartInputFile::initialize()
{
    readMagicNumberAndVersionField(*_data->is, _data->version);
    
    bool multipart = isMultiPart(_data->version);
    bool tiled = isTiled(_data->version);

    //
    // Multipart files don't have and shouldn't have the tiled bit set.
    //

    if (tiled && multipart)
        throw IEX_NAMESPACE::InputExc ("Multipart files cannot have the tiled bit set");

    
    int pos = 0;
    while (true)
    {
        Header header;
        header.readFrom(*_data->is, _data->version);

        //
        // If we read nothing then we stop reading.
        //

        if (header.readsNothing())
        {
            pos++;
            break;
        }

        _data->_headers.push_back(header);
        
        if(multipart == false)
          break;
    }

    //
    // Perform usual check on headers.
    //

    if ( _data->_headers.size() == 0)
    {
        throw IEX_NAMESPACE::ArgExc ("Files must contain at least one header");
    }

    for (size_t i = 0; i < _data->_headers.size(); i++)
    {
        //
        // Silently invent a type if the file is a single part regular image.
        //

        if( _data->_headers[i].hasType() == false )
        {
            if(multipart)

                throw IEX_NAMESPACE::ArgExc ("Every header in a multipart file should have a type");
          
            _data->_headers[i].setType(tiled ? TILEDIMAGE : SCANLINEIMAGE);
        }
        else
        {
            
            //
            // Silently fix the header type if it's wrong
            // (happens when a regular Image file written by EXR_2.0 is rewritten by an older library,
            //  so doesn't effect deep image types)
            //

            if(!multipart && !isNonImage(_data->version))
            {
                _data->_headers[i].setType(tiled ? TILEDIMAGE : SCANLINEIMAGE);
            }
        }
         

        
        if( _data->_headers[i].hasName() == false )
        {
            if(multipart)
                throw IEX_NAMESPACE::ArgExc ("Every header in a multipart file should have a name");
        }
        
        if (isTiled(_data->_headers[i].type()))
            _data->_headers[i].sanityCheck(true, multipart);
        else
            _data->_headers[i].sanityCheck(false, multipart);
    }

    //
    // Check name uniqueness.
    //

    if (multipart)
    {
        set<string> names;
        for (size_t i = 0; i < _data->_headers.size(); i++)
        {
        
            if (names.find(_data->_headers[i].name()) != names.end())
            {
                throw IEX_NAMESPACE::InputExc ("Header name " + _data->_headers[i].name() +
                                   " is not a unique name.");
            }
            names.insert(_data->_headers[i].name());
        }
    }
    
    //
    // Check shared attributes compliance.
    //

    if (multipart && strictSharedAttribute)
    {
        for (size_t i = 1; i < _data->_headers.size(); i++)
        {
            vector <string> attrs;
            if (_data->checkSharedAttributesValues (_data->_headers[0], _data->_headers[i], attrs))
            {
                string attrNames;
                for (size_t j=0; j<attrs.size(); j++)
                    attrNames += " " + attrs[j];
                throw IEX_NAMESPACE::InputExc ("Header name " + _data->_headers[i].name() +
                                     " has non-conforming shared attributes: "+
                                     attrNames);
            }
        }
    }

    //
    // Create InputParts and read chunk offset tables.
    //
        
    for (size_t i = 0; i < _data->_headers.size(); i++)
        _data->parts.push_back(
                new InputPartData(_data, _data->_headers[i], i, _data->numThreads, _data->version));

    _data->readChunkOffsetTables(_data->reconstructChunkOffsetTable);
}

TileOffsets*
MultiPartInputFile::Data::createTileOffsets(const Header& header)
{
    //
    // Get the dataWindow information
    //

    const Box2i &dataWindow = header.dataWindow();
    int minX = dataWindow.min.x;
    int maxX = dataWindow.max.x;
    int minY = dataWindow.min.y;
    int maxY = dataWindow.max.y;

    //
    // Precompute level and tile information
    //

    int* numXTiles = nullptr;
    int* numYTiles = nullptr;
    int numXLevels, numYLevels;
    TileDescription tileDesc = header.tileDescription();
    try
    {

        precalculateTileInfo (tileDesc,
                            minX, maxX,
                            minY, maxY,
                            numXTiles, numYTiles,
                            numXLevels, numYLevels);

        TileOffsets* tileOffsets = new TileOffsets (tileDesc.mode,
                                                    numXLevels,
                                                    numYLevels,
                                                    numXTiles,
                                                    numYTiles);
        delete [] numXTiles;
        delete [] numYTiles;

        return tileOffsets;

    }
    catch(...)
    {
        delete [] numXTiles;
        delete [] numYTiles;
        throw;
    }

}


void
MultiPartInputFile::Data::chunkOffsetReconstruction(OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is, const vector<InputPartData*>& parts)
{
    //
    // Reconstruct broken chunk offset tables. Stop once we received any exception.
    //

    uint64_t position = is.tellg();

    
    //
    // check we understand all the parts available: if not, we cannot continue
    // exceptions thrown here should trickle back up to the constructor
    //
    
    for (size_t i = 0; i < parts.size(); i++)
    {
        Header& header=parts[i]->header;
        
        //
        // do we have a valid type entry?
        // we only need them for true multipart files or single part non-image (deep) files
        //
        if(!header.hasType() && (isMultiPart(version) || isNonImage(version)))
        {
            throw IEX_NAMESPACE::ArgExc("cannot reconstruct incomplete file: part with missing type");
        }
        if(!isSupportedType(header.type()))
        {
            throw IEX_NAMESPACE::ArgExc("cannot reconstruct incomplete file: part with unknown type "+header.type());
        }
    }
    
    
    // how many chunks should we read? We should stop when we reach the end
    size_t total_chunks = 0;
        
    // for tiled-based parts, array of (pointers to) tileOffsets objects
    // to create mapping between tile coordinates and chunk table indices
    
    
    vector<TileOffsets*> tileOffsets(parts.size());
    
    // for scanline-based parts, number of scanlines in each chunk
    vector<int> rowsizes(parts.size());
        
    for(size_t i = 0 ; i < parts.size() ; i++)
    {
        total_chunks += parts[i]->chunkOffsets.size();
        if (isTiled(parts[i]->header.type()))
        {
            tileOffsets[i] = createTileOffsets(parts[i]->header);
        }else{
            tileOffsets[i] = NULL;
            // (TODO) fix this so that it doesn't need to be revised for future compression types.
            switch(parts[i]->header.compression())
            {
                case DWAB_COMPRESSION :
                    rowsizes[i] = 256;
                    break;
                case PIZ_COMPRESSION :
                case B44_COMPRESSION :
                case B44A_COMPRESSION :
                case DWAA_COMPRESSION :
                    rowsizes[i]=32;
                    break;
                case ZIP_COMPRESSION :
                case PXR24_COMPRESSION :
                    rowsizes[i]=16;
                    break;
                case ZIPS_COMPRESSION :
                case RLE_COMPRESSION :
                case NO_COMPRESSION :
                    rowsizes[i]=1;
                    break;
                default :
                    throw(IEX_NAMESPACE::ArgExc("Unknown compression method in chunk offset reconstruction"));
            }
        }
     }
        
     try
     {
            
        //
        // 
        //
        
        uint64_t chunk_start = position;
        for (size_t i = 0; i < total_chunks ; i++)
        {
            //
            // do we have a part number?
            //
            
            int partNumber = 0;
            if(isMultiPart(version))
            {
                OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, partNumber);
            }
            
            
            
            if(partNumber<0 || partNumber>= static_cast<int>(parts.size()))
            {
                throw IEX_NAMESPACE::IoExc("part number out of range");
            }
            
            Header& header = parts[partNumber]->header;

            // size of chunk NOT including multipart field
            
            uint64_t size_of_chunk=0;

            if (isTiled(header.type()))
            {
                //
                // 
                //
                int tilex,tiley,levelx,levely;
                OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, tilex);
                OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, tiley);
                OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, levelx);
                OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, levely);
                
                //std::cout << "chunk_start for " << tilex <<',' << tiley << ',' << levelx << ' ' << levely << ':' << chunk_start << std::endl;
                    
                
                if(!tileOffsets[partNumber])
                {
                    // this shouldn't actually happen - we should have allocated a valid
                    // tileOffsets for any part which isTiled
                    throw IEX_NAMESPACE::IoExc("part not tiled");
                    
                }
                
                if(!tileOffsets[partNumber]->isValidTile(tilex,tiley,levelx,levely))
                {
                    throw IEX_NAMESPACE::IoExc("invalid tile coordinates");
                }
                
                (*tileOffsets[partNumber])(tilex,tiley,levelx,levely)=chunk_start;
                
                // compute chunk sizes - different procedure for deep tiles and regular
                // ones
                if(header.type()==DEEPTILE)
                {
                    uint64_t packed_offset;
                    uint64_t packed_sample;
                    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, packed_offset);
                    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, packed_sample);
                    
                    //add 40 byte header to packed sizes (tile coordinates, packed sizes, unpacked size)
                    // check for bad values to prevent overflow
                    if ( (INT64_MAX - packed_offset < packed_sample) || ( INT64_MAX - (packed_offset + packed_sample) < 40ll) )
                    {
                          throw IEX_NAMESPACE::IoExc("Invalid chunk size");
                    }
                    size_of_chunk=packed_offset+packed_sample + 40ll;
                }
                else
                {
                    
                    // regular image has 20 bytes of header, 4 byte chunksize;
                    int chunksize;
                    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, chunksize);
                    // check for bad values to prevent overflow
                    if ( chunksize < 0 )
                    {
                          throw IEX_NAMESPACE::IoExc("Invalid chunk size");
                    }

                    size_of_chunk=static_cast<uint64_t>(chunksize) + 20ll;
                }
            }
            else
            {
                int y_coordinate;
                OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, y_coordinate);
                
                
                if(y_coordinate < header.dataWindow().min.y || y_coordinate > header.dataWindow().max.y)
                {
                   throw IEX_NAMESPACE::IoExc("y out of range");
                }
                y_coordinate -= header.dataWindow().min.y;
                y_coordinate /= rowsizes[partNumber];   
                
                if(y_coordinate < 0 || y_coordinate >= int(parts[partNumber]->chunkOffsets.size()))
                {
                   throw IEX_NAMESPACE::IoExc("chunk index out of range");
                }
                
                parts[partNumber]->chunkOffsets[y_coordinate]=chunk_start;
                
                if(header.type()==DEEPSCANLINE)
                {
                    uint64_t packed_offset;
                    uint64_t packed_sample;
                    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, packed_offset);
                    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, packed_sample);

                    // check for bad values to prevent overflow
                    if ( packed_offset < 0 ||
                        packed_sample < 0 ||
                        (INT64_MAX - packed_offset < packed_sample) ||
                        ( INT64_MAX - (packed_offset + packed_sample) < 28ll) )
                    {
                          throw IEX_NAMESPACE::IoExc("Invalid chunk size");
                    }
                    size_of_chunk=packed_offset+packed_sample + 28ll;
                }
                else
                {
                    int chunksize;
                    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, chunksize);   

                    // check for bad values to prevent overflow
                    if ( chunksize < 0 )
                    {
                          throw IEX_NAMESPACE::IoExc("Invalid chunk size");
                    }
                    size_of_chunk=static_cast<uint64_t>(chunksize) + 8ll;
                }
                
            }
            
            if(isMultiPart(version))
            {
                chunk_start+=4;
            }

            if (  (INT64_MAX - chunk_start) < size_of_chunk )
            {
               throw IEX_NAMESPACE::IoExc("File pointer overflow during reconstruction");
            }
            chunk_start+=size_of_chunk;
            
            is.seekg(chunk_start);
            
        }
        
    }
    catch (...) //NOSONAR - suppress vulnerability reports from SonarCloud.
    {
        //
        // Suppress all exceptions.  This functions is
        // called only to reconstruct the line offset
        // table for incomplete files, and exceptions
        // are likely.
        //
    }

    // copy tiled part data back to chunk offsets
    
    for(size_t partNumber=0;partNumber<parts.size();partNumber++)
    {
        if(tileOffsets[partNumber])
        {
            size_t pos=0;
            vector<vector<vector <uint64_t> > > offsets = tileOffsets[partNumber]->getOffsets();
            for (size_t l = 0; l < offsets.size(); l++)
                for (size_t y = 0; y < offsets[l].size(); y++)
                    for (size_t x = 0; x < offsets[l][y].size(); x++)
                    {
                        parts[ partNumber ]->chunkOffsets[pos] = offsets[l][y][x];
                        pos++;
                    }
           delete tileOffsets[partNumber];
        }
    }

    is.clear();
    is.seekg (position);
}

InputPartData*
MultiPartInputFile::Data::getPart(int partNumber)
{
    if (partNumber < 0 || partNumber >= (int) parts.size())
    {
            THROW ( IEX_NAMESPACE::ArgExc , "MultiPartInputFile::getPart called with invalid part " << partNumber << " on file with " << parts.size() << " parts");
    }
    return parts[partNumber];
}

namespace{
static const int gLargeChunkTableSize = 1024*1024;
}

void
MultiPartInputFile::Data::readChunkOffsetTables(bool reconstructChunkOffsetTable)
{
    bool brokenPartsExist = false;

    for (size_t i = 0; i < parts.size(); i++)
    {
        int chunkOffsetTableSize = getChunkOffsetTableSize(parts[i]->header);

        //
        // avoid allocating excessive memory.
        // If the chunktablesize claims to be large,
        // check the file is big enough to contain the table before allocating memory.
        // Attempt to read the last entry in the table. Either the seekg() or the read()
        // call will throw an exception if the file is too small to contain the table
        //
        if (chunkOffsetTableSize > gLargeChunkTableSize)
        {
            uint64_t pos = is->tellg();
            is->seekg(pos + (chunkOffsetTableSize-1)*sizeof(uint64_t));
            uint64_t temp;
            OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*is, temp);
            is->seekg(pos);

        }

        parts[i]->chunkOffsets.resize(chunkOffsetTableSize);



        for (int j = 0; j < chunkOffsetTableSize; j++)
            OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*is, parts[i]->chunkOffsets[j]);

        //
        // Check chunk offsets, reconstruct if broken.
        // At first we assume the table is complete.
        //
        parts[i]->completed = true;
        for (int j = 0; j < chunkOffsetTableSize; j++)
        {
            if (parts[i]->chunkOffsets[j] <= 0)
            {
                brokenPartsExist = true;
                parts[i]->completed = false;
                break;
            }
        }
    }

    if (brokenPartsExist && reconstructChunkOffsetTable)
        chunkOffsetReconstruction(*is, parts);
}

int 
MultiPartInputFile::version() const
{
    return _data->version;
}

bool 
MultiPartInputFile::partComplete(int part) const
{

    if(part<0 || part >= _data->_headers.size())
    {
        THROW ( IEX_NAMESPACE::ArgExc , "MultiPartInputFile::partComplete called with invalid part " << part << " on file with " << _data->_headers.size() << " parts");
    }
  return _data->parts[part]->completed;
}

int 
MultiPartInputFile::parts() const
{
   return int(_data->_headers.size());
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
