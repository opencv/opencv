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


#ifndef INCLUDED_IMF_DEEPCOMPOSITING_H
#define INCLUDED_IMF_DEEPCOMPOSITING_H

//-----------------------------------------------------------------------------
//
//	Class to sort and composite deep samples into a frame buffer
//      You may derive from this class to change the way that CompositeDeepScanLine
//      and CompositeDeepTile combine samples together - pass an instance of your derived
//      class to the compositing engine
//
//-----------------------------------------------------------------------------

#include "ImfForward.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class IMF_EXPORT DeepCompositing
{
    public:
        DeepCompositing();
        virtual ~DeepCompositing();
        
        
        //////////////////////////////////////////////
        ///
        /// composite together the given channels
        /// 
        ///  @param outputs       - return array of pixel values - 
        ///  @param inputs        - arrays of input sample
        ///  @param channel_names - array of channel names for corresponding channels
        ///  @param num_channels  - number of active channels (3 or greater)    
        ///  @param num_samples   - number of values in all input arrays
        ///  @param sources       - number of different sources
        ///
        /// each array input has num_channels entries: outputs[n] should be the composited
        /// values in array inputs[n], whose name will be given by channel_names[n]
        /// 
        /// The channel ordering shall be as follows:
        /// Position Channel
        ///    0     Z
        ///    1     ZBack (if no ZBack, then inputs[1]==inputs[0] and channel_names[1]==channel_names[0])
        ///    2     A (alpha channel)
        ///    3-n   other channels - only channels in the frame buffer will appear here
        ///
        /// since a Z and Alpha channel is required, and channel[1] is ZBack or another copy of Z
        /// there will always be 3 or more channels.
        ///
        /// The default implementation calls sort() if and only if more than one source is active,
        /// composites all samples together using the Over operator from front to back,
        /// stopping as soon as a sample with alpha=1 is found
        /// It also blanks all outputs if num_samples==0
        ///
        /// note - multiple threads may call composite_pixel simultaneously for different pixels
        ///
        ///
        //////////////////////////////////////////////
        virtual void composite_pixel(float outputs[],
                                     const float * inputs[],
                                     const char * channel_names[],
                                     int num_channels,
                                     int num_samples,
                                     int sources
                                     );
                                     
        
       
       ////////////////////////////////////////////////////////////////
       ///
       /// find the depth order for samples with given channel values
       /// does not sort the values in-place. Instead it populates
       /// array 'order' with the desired sorting order
       ///
       /// the default operation sorts samples from front to back according to their Z channel
       ///
       /// @param order         - required output order. order[n] shall be the nth closest sample
       /// @param inputs        - arrays of input samples, one array per channel_name
       /// @param channel_names - array of channel names for corresponding channels
       /// @param num_channels  - number of channels (3 or greater)  
       /// @param num_samples   - number of samples in each array
       /// @param sources       - number of different sources the data arises from
       ///
       /// the channel layout is identical to composite_pixel() 
       ///
       ///////////////////////////////////////////////////////////////
                                     
       virtual void sort(int order[],
                         const float * inputs[],
                         const char * channel_names[],
                         int num_channels,
                         int num_samples,
                         int sources);
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
