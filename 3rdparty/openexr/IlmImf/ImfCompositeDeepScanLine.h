///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012, Weta Digital Ltd
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Weta Digital nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


#ifndef INCLUDED_IMF_COMPOSITEDEEPSCANLINE_H
#define INCLUDED_IMF_COMPOSITEDEEPSCANLINE_H

//-----------------------------------------------------------------------------
//
//	Class to composite deep samples into a frame buffer
//      Initialise with a deep input part or deep inputfile
//      (also supports multiple files and parts, and will 
//       composite them together, as long as their sizes and channelmaps agree)
//       
//      Then call setFrameBuffer, and readPixels, exactly as for reading 
//      regular scanline images.
//
//      Restrictions - source file(s) must contain at least Z and alpha channels
//                   - if multiple files/parts are provided, sizes must match
//                   - all requested channels will be composited as premultiplied
//                   - only half and float channels can be requested
//
//      This object should not be considered threadsafe
//  
//      The default compositing engine will give spurious results with overlapping
//      volumetric samples - you may derive from DeepCompositing class, override the 
//      sort_pixel() and composite_pixel() functions, and pass an instance to 
//      setCompositing(). 
//
//-----------------------------------------------------------------------------

#include "ImfForward.h"
#include "ImfNamespace.h"
#include "ImfExport.h"
#include <ImathBox.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class CompositeDeepScanLine
{
    public:
        IMF_EXPORT
        CompositeDeepScanLine();
        IMF_EXPORT
        virtual ~CompositeDeepScanLine();
        
        /// set the source data as a part
        ///@note all parts must remain valid until after last interaction with DeepComp
        IMF_EXPORT
        void addSource(DeepScanLineInputPart * part);
        
        /// set the source data as a file
        ///@note all file must remain valid until after last interaction with DeepComp
        IMF_EXPORT
        void addSource(DeepScanLineInputFile * file);
        
        
        /////////////////////////////////////////
        //
        // set the frame buffer for output values
        // the buffers specified must be large enough
        // to handle the dataWindow() 
        //
        /////////////////////////////////////////
        IMF_EXPORT
        void setFrameBuffer(const FrameBuffer & fr);
        
        
        
        /////////////////////////////////////////
        //
        // retrieve frameBuffer
        //
        ////////////////////////////////////////
        IMF_EXPORT
        const FrameBuffer & frameBuffer() const;
        
        
        //////////////////////////////////////////////////
        //
        // read scanlines start to end from the source(s)
        // storing the result in the frame buffer provided
        //
        //////////////////////////////////////////////////
        
        IMF_EXPORT
        void readPixels(int start,int end);
        
        IMF_EXPORT
        int sources() const; // return number of sources
        
        /////////////////////////////////////////////////
        //
        // retrieve the datawindow
        // If multiple parts are specified, this will
        // be the union of the dataWindow of all parts
        //
        ////////////////////////////////////////////////
        
        IMF_EXPORT
        const IMATH_NAMESPACE::Box2i & dataWindow() const;
        
 
        //
        // override default sorting/compositing operation
        // (otherwise an instance of the base class will be used)
        //
        
        IMF_EXPORT
        void setCompositing(DeepCompositing *);
        
      struct Data; 
    private :  
      struct Data *_Data;
      
      CompositeDeepScanLine(const CompositeDeepScanLine &); // not implemented
      const CompositeDeepScanLine & operator=(const CompositeDeepScanLine &);  // not implemented
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
