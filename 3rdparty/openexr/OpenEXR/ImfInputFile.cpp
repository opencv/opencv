//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	class InputFile
//
//-----------------------------------------------------------------------------

#include "ImfInputFile.h"

#include "ImfCheckedArithmetic.h"

#include "ImfScanLineInputFile.h"
#include "ImfTiledInputFile.h"
#include "ImfChannelList.h"
#include "ImfInputPartData.h"
#include "ImfInputStreamMutex.h"
#include "ImfMisc.h"
#include "ImfStdIO.h"
#include "ImfVersion.h"
#include "ImfPartType.h"
#include "ImfInputPartData.h"
#include "ImfMultiPartInputFile.h"

#include "ImfCompositeDeepScanLine.h"
#include "ImfDeepScanLineInputFile.h"

#include <ImathFun.h>
#include "Iex.h"
#include <half.h>

#include <fstream>
#include <algorithm>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::divp;
using IMATH_NAMESPACE::modp;

//
// Struct InputFile::Data stores things that will be
// needed between calls to readPixels
//

struct InputFile::Data
#if ILMTHREAD_THREADING_ENABLED
    : public std::mutex
#endif
{
    Header              header;
    int                 version;
    bool                isTiled;

    TiledInputFile *	tFile;
    ScanLineInputFile *	sFile;
    DeepScanLineInputFile * dsFile;

    LineOrder		lineOrder;      // the file's lineorder
    int			minY;           // data window's min y coord
    int			maxY;           // data window's max x coord
    
    FrameBuffer		tFileBuffer; 
    FrameBuffer *	cachedBuffer;
    CompositeDeepScanLine * compositor; // for loading deep files
    
    int			cachedTileY;
    int                 offset;
    
    int                 numThreads;

    int                 partNumber;
    InputPartData*      part;

    bool                multiPartBackwardSupport;
    MultiPartInputFile* multiPartFile;
    InputStreamMutex    * _streamData;
    bool                _deleteStream;

     Data (int numThreads);
    ~Data ();

    Data (const Data& other) = delete;
    Data& operator = (const Data& other) = delete;
    Data (Data&& other) = delete;
    Data& operator = (Data&& other) = delete;

    void		deleteCachedBuffer();
};


InputFile::Data::Data (int numThreads):
    isTiled (false),
    tFile (0),
    sFile (0),
    dsFile(0),
    cachedBuffer (0),
    compositor(0),
    cachedTileY (-1),
    numThreads (numThreads),
    partNumber (-1),
    part(NULL),
    multiPartBackwardSupport (false),
    multiPartFile (0),
    _streamData(0),
    _deleteStream(false)
           
{
    // empty
}


InputFile::Data::~Data ()
{
    if (tFile)
        delete tFile;
    if (sFile)
        delete sFile;
    if (dsFile)
        delete dsFile;
    if (compositor)
        delete compositor;

    deleteCachedBuffer();

    if (multiPartBackwardSupport && multiPartFile)
        delete multiPartFile;
}


void	
InputFile::Data::deleteCachedBuffer()
{
    //
    // Delete the cached frame buffer, and all memory
    // allocated for the slices in the cached frameBuffer.
    //

    if (cachedBuffer)
    {
	for (FrameBuffer::Iterator k = cachedBuffer->begin();
	     k != cachedBuffer->end();
	     ++k)
	{
	    Slice &s = k.slice();

	    switch (s.type)
	    {
	      case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

		delete [] (((unsigned int *)s.base) + offset);
		break;

	      case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

		delete [] ((half *)s.base + offset);
		break;

	      case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

		delete [] (((float *)s.base) + offset);
		break;
              case NUM_PIXELTYPES :
                  throw(IEX_NAMESPACE::ArgExc("Invalid pixel type"));
	    }                
	}

	//
	// delete the cached frame buffer
	//

	delete cachedBuffer;
	cachedBuffer = 0;
    }
}


namespace {

void
bufferedReadPixels (InputFile::Data* ifd, int scanLine1, int scanLine2)
{
    //
    // bufferedReadPixels reads each row of tiles that intersect the
    // scan-line range (scanLine1 to scanLine2). The previous row of
    // tiles is cached in order to prevent redundent tile reads when
    // accessing scanlines sequentially.
    //

    int minY = std::min (scanLine1, scanLine2);
    int maxY = std::max (scanLine1, scanLine2);

    if (minY < ifd->minY || maxY >  ifd->maxY)
    {
        throw IEX_NAMESPACE::ArgExc ("Tried to read scan line outside "
			   "the image file's data window.");
    }

    //
    // The minimum and maximum y tile coordinates that intersect this
    // scanline range
    //

    int minDy = (minY - ifd->minY) / ifd->tFile->tileYSize();
    int maxDy = (maxY - ifd->minY) / ifd->tFile->tileYSize();

    //
    // Figure out which one is first in the file so we can read without seeking
    //

    int yStart, yEnd, yStep;

    if (ifd->lineOrder == DECREASING_Y)
    {
        yStart = maxDy;
        yEnd = minDy - 1;
        yStep = -1;
    }
    else
    {
        yStart = minDy;
        yEnd = maxDy + 1;
        yStep = 1;
    }

    //
    // the number of pixels in a row of tiles
    //

    Box2i levelRange = ifd->tFile->dataWindowForLevel(0);
    
    //
    // Read the tiles into our temporary framebuffer and copy them into
    // the user's buffer
    //

    for (int j = yStart; j != yEnd; j += yStep)
    {
        Box2i tileRange = ifd->tFile->dataWindowForTile (0, j, 0);

        int minYThisRow = std::max (minY, tileRange.min.y);
        int maxYThisRow = std::min (maxY, tileRange.max.y);

        if (j != ifd->cachedTileY)
        {
            //
            // We don't have any valid buffered info, so we need to read in
            // from the file.
            // if no channels are being read that are present in file, cachedBuffer will be empty
            //

            if (ifd->cachedBuffer && ifd->cachedBuffer->begin() != ifd->cachedBuffer->end())
            {
                ifd->tFile->readTiles (0, ifd->tFile->numXTiles (0) - 1, j, j);
            }

            ifd->cachedTileY = j;
        }

        //
        // Copy the data from our cached framebuffer into the user's
        // framebuffer.
        //

        for (FrameBuffer::ConstIterator k = ifd->tFileBuffer.begin();
             k != ifd->tFileBuffer.end();
             ++k)
        {


            Slice toSlice = k.slice();		// slice to read from
            char* toPtr;

            int xStart = levelRange.min.x;
            int yStart = minYThisRow;

            while (modp (xStart, toSlice.xSampling) != 0)
                ++xStart;

            while (modp (yStart, toSlice.ySampling) != 0)
                ++yStart;

            FrameBuffer::ConstIterator c = ifd->cachedBuffer->find(k.name());
            intptr_t toBase = reinterpret_cast<intptr_t>(toSlice.base);


            if( c!=ifd->cachedBuffer->end())
            {
                //
                // output channel was read from source image: copy to output slice
                //
                Slice fromSlice = c.slice();	// slice to write to
                intptr_t fromBase = reinterpret_cast<intptr_t>(fromSlice.base);

                int size = pixelTypeSize (toSlice.type);
                char* fromPtr;

                for (int y = yStart;
                    y <= maxYThisRow;
                    y += toSlice.ySampling)
                {
                    //
                    // Set the pointers to the start of the y scanline in
                    // this row of tiles
                    //

                    fromPtr = reinterpret_cast<char*> (fromBase  +
                            (y - tileRange.min.y) * fromSlice.yStride +
                            xStart * fromSlice.xStride);

                    toPtr = reinterpret_cast<char*> (toBase +
                            divp (y, toSlice.ySampling) * toSlice.yStride +
                            divp (xStart, toSlice.xSampling) * toSlice.xStride);

                    //
                    // Copy all pixels for the scanline in this row of tiles
                    //

                    for (int x = xStart;
                        x <= levelRange.max.x;
                        x += toSlice.xSampling)
                    {
                        for (int i = 0; i < size; ++i)
                            toPtr[i] = fromPtr[i];

                        fromPtr += fromSlice.xStride * toSlice.xSampling;
                        toPtr += toSlice.xStride;
                    }
                }
            }
            else
            {

                //
                // channel wasn't present in source file: fill output slice
                //
                for (int y = yStart;
                    y <= maxYThisRow;
                    y += toSlice.ySampling)
                {

                    toPtr = reinterpret_cast<char*> (toBase+
                            divp (y, toSlice.ySampling) * toSlice.yStride +
                            divp (xStart, toSlice.xSampling) * toSlice.xStride);

                    //
                    // Copy all pixels for the scanline in this row of tiles
                    //

                    switch ( toSlice.type)
                    {
                        case UINT:
                        {
                            unsigned int fill = toSlice.fillValue;
                            for (int x = xStart;
                                x <= levelRange.max.x;
                                x += toSlice.xSampling)
                            {
                                * reinterpret_cast<unsigned int*>(toPtr) = fill;
                                toPtr += toSlice.xStride;
                            }
                            break;
                        }
                        case HALF :
                        {
                            half fill = toSlice.fillValue;
                            for (int x = xStart;
                                x <= levelRange.max.x;
                                x += toSlice.xSampling)
                            {
                                * reinterpret_cast<half*>(toPtr) = fill;
                                toPtr += toSlice.xStride;
                            }
                            break;
                        }
                        case FLOAT :
                        {
                            float fill = toSlice.fillValue;
                            for (int x = xStart;
                                x <= levelRange.max.x;
                                x += toSlice.xSampling)
                            {
                                * reinterpret_cast<float*>(toPtr) = fill;
                                toPtr += toSlice.xStride;
                            }
                            break;
                        }
                        case NUM_PIXELTYPES :
                        {
                            break;
                        }

                    }
                }
            }
        }
    }
}

} // namespace



InputFile::InputFile (const char fileName[], int numThreads):
    _data (new Data (numThreads))
{
    _data->_streamData = NULL;
    _data->_deleteStream=true;
    
    OPENEXR_IMF_INTERNAL_NAMESPACE::IStream* is = 0;
    try
    {
        is = new StdIFStream (fileName);
        readMagicNumberAndVersionField(*is, _data->version);

        //
        // compatibility to read multipart file.
        //
        if (isMultiPart(_data->version))
        {
            compatibilityInitialize(*is);
        }
        else
        {
            _data->_streamData = new InputStreamMutex();
            _data->_streamData->is = is;
            _data->header.readFrom (*_data->_streamData->is, _data->version);
            
            if(isNonImage(_data->version))
            {
                if(!_data->header.hasType())
                {
                      throw(IEX_NAMESPACE::InputExc("Non-image files must have a 'type' attribute"));
                }
            }

            // fix type attribute in single part regular image types
            // (may be wrong if an old version of OpenEXR converts
            // a tiled image to scanline or vice versa)
            if(!isNonImage(_data->version)  && 
               !isMultiPart(_data->version) && 
               _data->header.hasType())
            {
                _data->header.setType(isTiled(_data->version) ? TILEDIMAGE : SCANLINEIMAGE);
            }
            
            _data->header.sanityCheck (isTiled (_data->version));

            initialize();
        }
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        if (is)          delete is;
         
        if ( _data && !_data->multiPartBackwardSupport  && _data->_streamData)
        {
            delete _data->_streamData;
            _data->_streamData=NULL;
        }
        
        if (_data)       delete _data;
        _data=NULL;

        REPLACE_EXC (e, "Cannot read image file "
                     "\"" << fileName << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        if (is)          delete is;
        if (_data && !_data->multiPartBackwardSupport && _data->_streamData)
        {
            delete _data->_streamData;
        }
        if (_data)       delete _data;

        throw;
    }
}


InputFile::InputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int numThreads):
    _data (new Data (numThreads))
{
    _data->_streamData=NULL;
    _data->_deleteStream=false;
    try
    {
        readMagicNumberAndVersionField(is, _data->version);

        //
        // Backward compatibility to read multpart file.
        //
        if (isMultiPart(_data->version))
        {
            compatibilityInitialize(is);
        }
        else
        {
            _data->_streamData = new InputStreamMutex();
            _data->_streamData->is = &is;
            _data->header.readFrom (*_data->_streamData->is, _data->version);
            
            if(isNonImage(_data->version))
            {
                if(!_data->header.hasType())
                {
                      throw(IEX_NAMESPACE::InputExc("Non-image files must have a 'type' attribute"));
                }
            }

            // fix type attribute in single part regular image types
            // (may be wrong if an old version of OpenEXR converts
            // a tiled image to scanline or vice versa)
            if(!isNonImage(_data->version)  && 
               !isMultiPart(_data->version) &&  
               _data->header.hasType())
            {
                _data->header.setType(isTiled(_data->version) ? TILEDIMAGE : SCANLINEIMAGE);
            }
            
            _data->header.sanityCheck (isTiled (_data->version));

            initialize();
        }
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        if (_data && !_data->multiPartBackwardSupport && _data->_streamData) delete _data->_streamData;
        if (_data)       delete _data;
        _data=NULL; 

        REPLACE_EXC (e, "Cannot read image file "
                     "\"" << is.fileName() << "\". " << e.what());
        throw;
    }
    catch (...)
    {
        if (_data &&  !_data->multiPartBackwardSupport  && _data->_streamData) delete _data->_streamData;
        if (_data)       delete _data;
        _data=NULL;
        throw;
    }
}


InputFile::InputFile (InputPartData* part) :
    _data (new Data (part->numThreads))
{
    _data->_deleteStream=false;
    try
    {
       multiPartInitialize (part);
    }
    catch(...)
    {
        delete _data;
        throw;
    }
}


void
InputFile::compatibilityInitialize (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is)
{
    is.seekg(0);

    //
    // Construct a MultiPartInputFile, initialize InputFile
    // with the part 0 data.
    // (TODO) may want to have a way to set the reconstruction flag.
    //
    _data->multiPartBackwardSupport = true;
    _data->multiPartFile = new MultiPartInputFile(is, _data->numThreads);
    InputPartData* part = _data->multiPartFile->getPart(0);

    multiPartInitialize (part);
}


void
InputFile::multiPartInitialize (InputPartData* part)
{
    _data->_streamData = part->mutex;
    _data->version = part->version;
    _data->header = part->header;
    _data->partNumber = part->partNumber;
    _data->part = part;

    initialize();
}


void
InputFile::initialize ()
{
    if (!_data->part)
    {
        if(_data->header.hasType() && _data->header.type()==DEEPSCANLINE)
        {
            _data->isTiled=false;
            const Box2i &dataWindow = _data->header.dataWindow();
            _data->minY = dataWindow.min.y;
            _data->maxY = dataWindow.max.y;
            
            _data->dsFile = new DeepScanLineInputFile (_data->header,
                                               _data->_streamData->is,
                                               _data->version,
                                               _data->numThreads);
            _data->compositor = new CompositeDeepScanLine;
            _data->compositor->addSource(_data->dsFile);
        }
        
        else if (isTiled (_data->version) && !isNonImage(_data->version)) 
        {
            _data->isTiled = true;
            _data->lineOrder = _data->header.lineOrder();

            //
            // Save the dataWindow information
            //
    
            const Box2i &dataWindow = _data->header.dataWindow();
            _data->minY = dataWindow.min.y;
            _data->maxY = dataWindow.max.y;

            _data->tFile = new TiledInputFile (_data->header,
                                               _data->_streamData->is,
                                               _data->version,
                                               _data->numThreads);
        }
        
        else if(!_data->header.hasType() || _data->header.type()==SCANLINEIMAGE)
        {
            _data->sFile = new ScanLineInputFile (_data->header,
                                                  _data->_streamData->is,
                                                  _data->numThreads);
        }else{
            // type set but not recognised
            
            THROW(IEX_NAMESPACE::ArgExc, "InputFile cannot handle parts of type " << _data->header.type());
        }
    }
    else
    {
        if(_data->header.hasType() && _data->header.type()==DEEPSCANLINE)
        {
            _data->isTiled=false;
            const Box2i &dataWindow = _data->header.dataWindow();
            _data->minY = dataWindow.min.y;
            _data->maxY = dataWindow.max.y;
            
            _data->dsFile = new DeepScanLineInputFile (_data->part);
            _data->compositor = new CompositeDeepScanLine;
            _data->compositor->addSource(_data->dsFile);
        }
        else if (_data->header.hasType() && _data->header.type()==TILEDIMAGE)
        {
            _data->isTiled = true;
            _data->lineOrder = _data->header.lineOrder();

            //
            // Save the dataWindow information
            //

            const Box2i &dataWindow = _data->header.dataWindow();
            _data->minY = dataWindow.min.y;
            _data->maxY = dataWindow.max.y;

            _data->tFile = new TiledInputFile (_data->part);
        }
        else if(!_data->header.hasType() || _data->header.type()==SCANLINEIMAGE)
        {
            _data->sFile = new ScanLineInputFile (_data->part);
        }else{
            THROW(IEX_NAMESPACE::ArgExc, "InputFile cannot handle parts of type " << _data->header.type());
            
        }
    }
}

#include <iostream>
InputFile::~InputFile ()
{
    if (_data->_deleteStream)
        delete _data->_streamData->is;

    // unless this file was opened via the multipart API,
    // delete the streamData object too
    if (_data->partNumber==-1 && _data->_streamData)
        delete _data->_streamData;

    if (_data)  delete _data;
}

const char *
InputFile::fileName () const
{
    return _data->_streamData->is->fileName();
}


const Header &
InputFile::header () const
{
    return _data->header;
}


int
InputFile::version () const
{
    return _data->version;
}


void
InputFile::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    if (_data->isTiled)
    {
#if ILMTHREAD_THREADING_ENABLED
        std::lock_guard<std::mutex> lock (*_data);
#endif
	//
        // We must invalidate the cached buffer if the new frame
	// buffer has a different set of channels than the old
	// frame buffer, or if the type of a channel has changed.
	//

	const FrameBuffer &oldFrameBuffer = _data->tFileBuffer;

	FrameBuffer::ConstIterator i = oldFrameBuffer.begin();
	FrameBuffer::ConstIterator j = frameBuffer.begin();

	while (i != oldFrameBuffer.end() && j != frameBuffer.end())
	{
	    if (strcmp (i.name(), j.name()) || i.slice().type != j.slice().type)
		break;

	    ++i;
	    ++j;
	}

	if (i != oldFrameBuffer.end() || j != frameBuffer.end())
        {
	    //
	    // Invalidate the cached buffer.
	    //

            _data->deleteCachedBuffer ();
	    _data->cachedTileY = -1;

	    //
	    // Create new a cached frame buffer.  It can hold a single
	    // row of tiles.  The cached buffer can be reused for each
	    // row of tiles because we set the yTileCoords parameter of
	    // each Slice to true.
	    //

	    const Box2i &dataWindow = _data->header.dataWindow();
	    _data->cachedBuffer = new FrameBuffer();
	    _data->offset = dataWindow.min.x;
	    
	    unsigned int tileRowSize =
                uiMult(dataWindow.max.x - dataWindow.min.x + 1U,
                       _data->tFile->tileYSize());

	    for (FrameBuffer::ConstIterator k = frameBuffer.begin();
		 k != frameBuffer.end();
		 ++k)
	    {
		Slice s = k.slice();

                //
                // omit adding channels that are not listed - 'fill' channels are added later
                //
                if ( _data->header.channels().find(k.name()) != _data->header.channels().end() )
                {
                    switch (s.type)
                    {
                    case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                        _data->cachedBuffer->insert
                            (k.name(),
                            Slice (UINT,
                                    (char *)(new unsigned int[tileRowSize] -
                                            _data->offset),
                                    sizeof (unsigned int),
                                    sizeof (unsigned int) *
                                        _data->tFile->levelWidth(0),
                                    1, 1,
                                    s.fillValue,
                                    false, true));
                        break;

                    case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

                        _data->cachedBuffer->insert
                            (k.name(),
                            Slice (HALF,
                                    (char *)(new half[tileRowSize] -
                                            _data->offset),
                                    sizeof (half),
                                    sizeof (half) *
                                        _data->tFile->levelWidth(0),
                                    1, 1,
                                    s.fillValue,
                                    false, true));
                        break;

                    case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                        _data->cachedBuffer->insert
                            (k.name(),
                            Slice (OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT,
                                    (char *)(new float[tileRowSize] -
                                            _data->offset),
                                    sizeof(float),
                                    sizeof(float) *
                                        _data->tFile->levelWidth(0),
                                    1, 1,
                                    s.fillValue,
                                    false, true));
                        break;

                    default:

                        throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
                    }
                }
	    }

	    _data->tFile->setFrameBuffer (*_data->cachedBuffer);

        }

	_data->tFileBuffer = frameBuffer;
    }
    else if(_data->compositor)
    {
        _data->compositor->setFrameBuffer(frameBuffer);
    }else {
        _data->sFile->setFrameBuffer(frameBuffer);
        _data->tFileBuffer = frameBuffer;
    }
}


const FrameBuffer &
InputFile::frameBuffer () const
{
    if(_data->compositor)
    {
        return _data->compositor->frameBuffer();
    }
    else if(_data->isTiled)
    {
#if ILMTHREAD_THREADING_ENABLED
        std::lock_guard<std::mutex> lock (*_data);
#endif
        return _data->tFileBuffer;
    }
    else
    {
	return _data->sFile->frameBuffer();
    }
}


bool
InputFile::isComplete () const
{
    if (_data->dsFile)
        return _data->dsFile->isComplete();
    else if (_data->isTiled)
	return _data->tFile->isComplete();
    else
	return _data->sFile->isComplete();
}

bool
InputFile::isOptimizationEnabled() const
{
   if(_data->sFile)
   {
       return _data->sFile->isOptimizationEnabled();
   }else{
       return false;
   }
}


void
InputFile::readPixels (int scanLine1, int scanLine2)
{
    if (_data->compositor)
    {
        _data->compositor->readPixels(scanLine1,scanLine2);
    }
    else if (_data->isTiled)
    {
#if ILMTHREAD_THREADING_ENABLED
        std::lock_guard<std::mutex> lock (*_data);
#endif
        bufferedReadPixels (_data, scanLine1, scanLine2);
    }
    else
    {
        _data->sFile->readPixels (scanLine1, scanLine2);
    }
}


void
InputFile::readPixels (int scanLine)
{
    readPixels (scanLine, scanLine);
}


void
InputFile::rawPixelData (int firstScanLine,
			 const char *&pixelData,
			 int &pixelDataSize)
{
    try
    {
        if (_data->dsFile)
        {
            throw IEX_NAMESPACE::ArgExc ("Tried to read a raw scanline "
            "from a deep image.");
        }
        
	else if (_data->isTiled)
	{
	    throw IEX_NAMESPACE::ArgExc ("Tried to read a raw scanline "
			       "from a tiled image.");
	}
        
        _data->sFile->rawPixelData (firstScanLine, pixelData, pixelDataSize);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error reading pixel data from image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}




void
InputFile::rawPixelDataToBuffer (int scanLine,
                                 char *pixelData,
                                 int &pixelDataSize) const
{
    try
    {
        if (_data->dsFile)
        {
            throw IEX_NAMESPACE::ArgExc ("Tried to read a raw scanline "
                                         "from a deep image.");
        }
        
        else if (_data->isTiled)
        {
            throw IEX_NAMESPACE::ArgExc ("Tried to read a raw scanline "
                                         "from a tiled image.");
        }
        
        _data->sFile->rawPixelDataToBuffer(scanLine, pixelData, pixelDataSize);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Error reading pixel data from image "
                     "file \"" << fileName() << "\". " << e.what());
        throw;
    }
}



void
InputFile::rawTileData (int &dx, int &dy,
			int &lx, int &ly,
			const char *&pixelData,
			int &pixelDataSize)
{
    try
    {
	if (!_data->isTiled)
	{
	    throw IEX_NAMESPACE::ArgExc ("Tried to read a raw tile "
			       "from a scanline-based image.");
	}
        
        _data->tFile->rawTileData (dx, dy, lx, ly, pixelData, pixelDataSize);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error reading tile data from image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


TiledInputFile*
InputFile::tFile()
{
    if (!_data->isTiled)
    {
	throw IEX_NAMESPACE::ArgExc ("Cannot get a TiledInputFile pointer "
			   "from an InputFile that is not tiled.");
    }

    return _data->tFile;
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
