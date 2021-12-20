//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_PIZ_COMPRESSOR_H
#define INCLUDED_IMF_PIZ_COMPRESSOR_H

//-----------------------------------------------------------------------------
//
//	class PizCompressor -- uses Wavelet and Huffman encoding.
//
//-----------------------------------------------------------------------------

#include "ImfNamespace.h"

#include "ImfCompressor.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class PizCompressor: public Compressor
{
  public:

    PizCompressor (const Header &hdr,
                   size_t maxScanLineSize,
                   size_t numScanLines);

    virtual ~PizCompressor ();

    PizCompressor (const PizCompressor& other) = delete;
    PizCompressor& operator = (const PizCompressor& other) = delete;
    PizCompressor (PizCompressor&& other) = delete;
    PizCompressor& operator = (PizCompressor&& other) = delete;

    virtual int		numScanLines () const;

    virtual Format	format () const;

    virtual int		compress (const char *inPtr,
				  int inSize,
				  int minY,
				  const char *&outPtr);                  
                  
    virtual int		compressTile (const char *inPtr,
				      int inSize,
				      IMATH_NAMESPACE::Box2i range,
				      const char *&outPtr);

    virtual int		uncompress (const char *inPtr,
				    int inSize,
				    int minY,
				    const char *&outPtr);
                    
    virtual int		uncompressTile (const char *inPtr,
					int inSize,
					IMATH_NAMESPACE::Box2i range,
					const char *&outPtr);
  private:

    struct ChannelData;
    
    int			compress (const char *inPtr,
				  int inSize,
				  IMATH_NAMESPACE::Box2i range,
				  const char *&outPtr);
 
    int			uncompress (const char *inPtr,
				    int inSize,
				    IMATH_NAMESPACE::Box2i range,
				    const char *&outPtr);

    int			_maxScanLineSize;
    Format		_format;
    int			_numScanLines;
    unsigned short *	_tmpBuffer;
    char *		_outBuffer;
    int			_numChans;
    const ChannelList &	_channels;
    ChannelData *	_channelData;
    int			_minX;
    int			_maxX;
    int			_maxY;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
