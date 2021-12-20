//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include "ImfMultiPartOutputFile.h"
#include "ImfBoxAttribute.h"
#include "ImfFloatAttribute.h"
#include "ImfTimeCodeAttribute.h"
#include "ImfChromaticitiesAttribute.h"
#include "ImfOutputPartData.h"
#include "ImfPartType.h"
#include "ImfOutputFile.h"
#include "ImfTiledOutputFile.h"
#include "ImfThreading.h"
#include "ImfMisc.h"
#include "ImfStdIO.h"
#include "ImfDeepScanLineOutputFile.h"
#include "ImfDeepTiledOutputFile.h"
#include "ImfOutputStreamMutex.h"

#include "ImfNamespace.h"
#include <Iex.h>


#include <set>


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;

using std::vector;
using std::map;
using std::set;


struct MultiPartOutputFile::Data: public OutputStreamMutex
{
        vector<OutputPartData*>         parts;        // Contains data to initialize Output files.
        bool                            deleteStream; // If we should delete the stream when destruction.
        int                             numThreads;   // The number of threads.
        std::map<int, GenericOutputFile*>    _outputFiles;
        std::vector<Header>                  _headers;
        
        
        void                    headerNameUniquenessCheck (const std::vector<Header> &headers);
        
        void                    writeHeadersToFile (const std::vector<Header> &headers); 
        
        void                    writeChunkTableOffsets (std::vector<OutputPartData*> &parts);
        
        
        //-------------------------------------
        // ensure that _headers is valid: called by constructors
        //-------------------------------------
        void                    do_header_sanity_checks(bool overrideSharedAttributes);
        
        // ------------------------------------------------
        // Given a source header, we copy over all the 'shared attributes' to
        // the destination header and remove any conflicting ones.
        // ------------------------------------------------
        void                    overrideSharedAttributesValues (const Header & src,
                                                                Header & dst);
                                                                
        // ------------------------------------------------
        // Given a source header, we check the destination header for any
        // attributes that are part of the shared attribute set. For attributes
        // present in both we check the values. For attribute present in 
        // destination but absent in source we return false.
        // For attributes present in src but missing from dst we return false
        // and add the attribute to dst.
        // We return false for all other cases.
        // If we return true then we also populate the conflictingAttributes
        // vector with the names of the attributes that failed the above.
        // ------------------------------------------------
        bool                    checkSharedAttributesValues (const Header & src,
                                                             const Header & dst, 
                                                             std::vector<std::string> & conflictingAttributes) const;
        Data (bool deleteStream, int numThreads):
            OutputStreamMutex(),
            deleteStream (deleteStream),
            numThreads (numThreads)
        {
        }
        

        ~Data()
        {
            if (deleteStream) delete os;

            for (size_t i = 0; i < parts.size(); i++)
                delete parts[i];
        }

    Data (const Data& other) = delete;
    Data& operator = (const Data& other) = delete;
    Data (Data&& other) = delete;
    Data& operator = (Data&& other) = delete;
    
};

void
MultiPartOutputFile::Data::do_header_sanity_checks(bool overrideSharedAttributes)
{
    size_t parts = _headers.size();
    if (parts == 0) 
        throw IEX_NAMESPACE::ArgExc ("Empty header list.");
    
    bool isMultiPart = (parts > 1); 
    
    //
    // Do part 0 checks first.
    //
    
    _headers[0].sanityCheck (_headers[0].hasTileDescription(), isMultiPart);
        
    
    if (isMultiPart)
    {
        // multipart files must contain a chunkCount attribute
        _headers[0].setChunkCount(getChunkOffsetTableSize(_headers[0]));
        
        for (size_t i = 1; i < parts; i++)
        {
            if (_headers[i].hasType() == false)
                throw IEX_NAMESPACE::ArgExc ("Every header in a multipart file should have a type");
            
            
            _headers[i].setChunkCount(getChunkOffsetTableSize(_headers[i]));
            _headers[i].sanityCheck (_headers[i].hasTileDescription(), isMultiPart);
            
            
            if (overrideSharedAttributes)
                overrideSharedAttributesValues(_headers[0],_headers[i]);
            else
            {
                std::vector<std::string> conflictingAttributes;
                bool valid =checkSharedAttributesValues (_headers[0],
                                                         _headers[i], 
                                                         conflictingAttributes);
                if (valid)
                {
                    string excMsg("Conflicting attributes found for header :: ");
                    excMsg += _headers[i].name();
                    for (size_t i=0; i<conflictingAttributes.size(); i++)
                        excMsg += " '" + conflictingAttributes[i] + "' ";
                                                             
                    THROW (IEX_NAMESPACE::ArgExc, excMsg);
                }
            }
        }
        
        headerNameUniquenessCheck(_headers);
        
    }else{
        
        // add chunk count offset to single part data (if not an image)
        
        if (_headers[0].hasType() && isImage(_headers[0].type()) == false)
        {
            _headers[0].setChunkCount(getChunkOffsetTableSize(_headers[0]));
        }
        
    }
}

    
MultiPartOutputFile::MultiPartOutputFile (const char fileName[],
                                          const Header * headers,
                                          int parts,
                                          bool overrideSharedAttributes,
                                          int numThreads)
:
    _data (new Data (true, numThreads))
{
    // grab headers
    _data->_headers.resize(parts);
    
    for(int i=0;i<parts;i++)
    {
       _data->_headers[i]=headers[i];
    }
    try
    {
  
         _data->do_header_sanity_checks(overrideSharedAttributes);

        //
        // Build parts and write headers and offset tables to file.
        //

        _data->os = new StdOFStream (fileName);
        for (size_t i = 0; i < _data->_headers.size(); i++)
            _data->parts.push_back( new OutputPartData(_data, _data->_headers[i], i, numThreads, parts>1 ) );

        writeMagicNumberAndVersionField(*_data->os, &_data->_headers[0],_data->_headers.size());
        _data->writeHeadersToFile(_data->_headers);
        _data->writeChunkTableOffsets(_data->parts);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _data;

        REPLACE_EXC (e, "Cannot open image file "
                     "\"" << fileName << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        delete _data;
        throw;
    }
}

MultiPartOutputFile::MultiPartOutputFile(OStream& os, 
                                         const Header* headers, 
                                         int parts, 
                                         bool overrideSharedAttributes, 
                                         int numThreads): 
                                         _data(new Data(false,numThreads))
{
    // grab headers
    _data->_headers.resize(parts);
    _data->os=&os;
    
    for(int i=0;i<parts;i++)
    {
        _data->_headers[i]=headers[i];
    }
    try
    {
        
        _data->do_header_sanity_checks(overrideSharedAttributes);
        
        //
        // Build parts and write headers and offset tables to file.
        //
        
        for (size_t i = 0; i < _data->_headers.size(); i++)
            _data->parts.push_back( new OutputPartData(_data, _data->_headers[i], i, numThreads, parts>1 ) );
        
        writeMagicNumberAndVersionField(*_data->os, &_data->_headers[0],_data->_headers.size());
        _data->writeHeadersToFile(_data->_headers);
        _data->writeChunkTableOffsets(_data->parts);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        delete _data;
        
        REPLACE_EXC (e, "Cannot open image stream "
                     "\"" << os.fileName() << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        delete _data;
        throw;
    }
}


const Header &
MultiPartOutputFile::header(int n) const
{
    if(n<0 || n >= int(_data->_headers.size()))
    {
        THROW ( IEX_NAMESPACE::ArgExc , "MultiPartOutputFile::header called with invalid part number " << n << " on file with " << _data->_headers.size() << " parts");
    }
    return _data->_headers[n];
}

int
MultiPartOutputFile::parts() const
{
   return _data->_headers.size();
}


MultiPartOutputFile::~MultiPartOutputFile ()
{
    for (map<int, GenericOutputFile*>::iterator it = _data->_outputFiles.begin();
         it != _data->_outputFiles.end(); it++)
    {
        delete it->second;
    }

    delete _data;
}

template <class T>
T*
MultiPartOutputFile::getOutputPart(int partNumber)
{

    if(partNumber<0 || partNumber >= int(_data->_headers.size()))
    {
        THROW ( IEX_NAMESPACE::ArgExc , "MultiPartOutputFile::getOutputPart called with invalid part number  " << partNumber << " on file with " << _data->_headers.size() << " parts");
    }

#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock(*_data);
#endif
    if (_data->_outputFiles.find(partNumber) == _data->_outputFiles.end())
    {
        T* file = new T(_data->parts[partNumber]);
        _data->_outputFiles.insert(std::make_pair(partNumber, (GenericOutputFile*) file));
        return file;
    }
    else return (T*) _data->_outputFiles[partNumber];
}

// instance above function for all four types
template OutputFile* MultiPartOutputFile::getOutputPart<OutputFile>(int);
template TiledOutputFile * MultiPartOutputFile::getOutputPart<TiledOutputFile>(int);
template DeepScanLineOutputFile * MultiPartOutputFile::getOutputPart<DeepScanLineOutputFile> (int);
template DeepTiledOutputFile * MultiPartOutputFile::getOutputPart<DeepTiledOutputFile> (int);



void 
MultiPartOutputFile::Data::overrideSharedAttributesValues(const Header & src, Header & dst)
{
    //
    // Display Window
    //
    const Box2iAttribute * displayWindow = 
    src.findTypedAttribute<Box2iAttribute> ("displayWindow");
    
    if (displayWindow)
        dst.insert ("displayWindow", *displayWindow);
    else 
        dst.erase ("displayWindow");
    
    
    //
    // Pixel Aspect Ratio
    //
    const FloatAttribute * pixelAspectRatio = 
    src.findTypedAttribute<FloatAttribute> ("pixelAspectRatio");
    
    if (pixelAspectRatio)
        dst.insert ("pixelAspectRatio", *pixelAspectRatio);
    else 
        dst.erase ("pixelAspectRatio");
    
    
    //
    // Timecode
    //
    const TimeCodeAttribute * timeCode = 
    src.findTypedAttribute<TimeCodeAttribute> ("timecode");
    
    if (timeCode)
        dst.insert ("timecode", *timeCode);
    else 
        dst.erase ("timecode");
    
    
    //
    // Chromaticities
    //
    const ChromaticitiesAttribute * chromaticities = 
    src.findTypedAttribute<ChromaticitiesAttribute> ("chromaticities");
    
    if (chromaticities)
        dst.insert ("chromaticities", *chromaticities);
    else 
        dst.erase ("chromaticities");
    
}


bool 
MultiPartOutputFile::Data::checkSharedAttributesValues(const Header & src,
        const Header & dst,
        vector<string> & conflictingAttributes) const
{
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
        if ((srcTimeCode && (srcTimeCode->value() != dstTimeCode->value())) ||
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
MultiPartOutputFile::Data::headerNameUniquenessCheck (const vector<Header> &headers)
{
    set<string> names;
    for (size_t i = 0; i < headers.size(); i++)
    {
        if (names.find(headers[i].name()) != names.end())
            throw IEX_NAMESPACE::ArgExc ("Each part should have a unique name.");
        names.insert(headers[i].name());
    }
}

void
MultiPartOutputFile::Data::writeHeadersToFile (const vector<Header> &headers)
{
    for (size_t i = 0; i < headers.size(); i++)
    {

        // (TODO) consider deep files' preview images here.
        if (headers[i].type() == TILEDIMAGE)
            parts[i]->previewPosition = headers[i].writeTo(*os, true);
        else
            parts[i]->previewPosition = headers[i].writeTo(*os, false);
    }

    //
    // If a multipart file, write zero-length attribute name to mark the end of all headers.
    //

    if (headers.size() !=1)
         OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*os, "");
}

void
MultiPartOutputFile::Data::writeChunkTableOffsets (vector<OutputPartData*> &parts)
{
    for (size_t i = 0; i < parts.size(); i++)
    {
        int chunkTableSize = getChunkOffsetTableSize(parts[i]->header);

        uint64_t pos = os->tellp();

        if (pos == static_cast<uint64_t>(-1))
            IEX_NAMESPACE::throwErrnoExc ("Cannot determine current file position (%T).");

        parts[i]->chunkOffsetTablePosition = os->tellp();

        //
        // Fill in empty data for now. We'll write actual offsets during destruction.
        //

        for (int j = 0; j < chunkTableSize; j++)
        {
            uint64_t empty = 0;
            OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (*os, empty);
        }
    }
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
