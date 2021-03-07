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


#include "ImfCompositeDeepScanLine.h"
#include "ImfDeepScanLineInputPart.h"
#include "ImfDeepScanLineInputFile.h"
#include "ImfChannelList.h"
#include "ImfFrameBuffer.h"
#include "ImfDeepFrameBuffer.h"
#include "ImfDeepCompositing.h"
#include "ImfPixelType.h"
#include "IlmThreadPool.h"

#include <Iex.h>
#include <vector>
OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using std::vector;
using std::string;
using IMATH_NAMESPACE::Box2i;
using ILMTHREAD_NAMESPACE::Task;
using ILMTHREAD_NAMESPACE::TaskGroup;
using ILMTHREAD_NAMESPACE::ThreadPool;



struct CompositeDeepScanLine::Data{
    public :
    vector<DeepScanLineInputFile *>     _file;   // array of files    
    vector<DeepScanLineInputPart *>     _part;   // array of parts 
    FrameBuffer            _outputFrameBuffer;   // output frame buffer provided
    bool                               _zback;   // true if we are using zback (otherwise channel 1 = channel 0)
    vector< vector<float> >      _channeldata;   // pixel values, read from the input, one array per channel
    vector< int >               _sampleCounts;   // total per-pixel sample counts,   
    Box2i                         _dataWindow;   // data window of combined inputs
    DeepCompositing *                   _comp;   // user-provided compositor
    vector<string>                  _channels;   // names of channels that will be composited
    vector<int>                    _bufferMap;   // entry _outputFrameBuffer[n].name() == _channels[ _bufferMap[n] ].name()
    
    void check_valid(const Header & header);     // check newly added part/file is OK; on first good call, set _zback/_dataWindow

    //
    // set up the given deep frame buffer to contain the required channels
    // resize counts and pointers to the width of _dataWindow
    // zero-out all counts, since the datawindow may be smaller than/not include this part
    //

    void handleDeepFrameBuffer (DeepFrameBuffer & buf,
                                vector<unsigned int> & counts,        //per-pixel counts
                                vector< vector<float *> > & pointers, //per-channel-per-pixel pointers to data
                                const Header & header,
                                int start,
                                int end);

    Data();
};

CompositeDeepScanLine::Data::Data() : _zback(false) , _comp(NULL) {}

CompositeDeepScanLine::CompositeDeepScanLine() : _Data(new Data) {}

CompositeDeepScanLine::~CompositeDeepScanLine()
{
   delete _Data;
}

void
CompositeDeepScanLine::addSource(DeepScanLineInputPart* part)
{
  _Data->check_valid(part->header());
  _Data->_part.push_back(part);
}

void
CompositeDeepScanLine::addSource(DeepScanLineInputFile* file)
{
    _Data->check_valid(file->header());
    _Data->_file.push_back(file);
}

int 
CompositeDeepScanLine::sources() const
{
   return int(_Data->_part.size())+int(_Data->_file.size());
}

void
CompositeDeepScanLine::Data::check_valid(const Header & header)
{

    bool has_z=false;
    bool has_alpha=false;
    // check good channel names
    for( ChannelList::ConstIterator i=header.channels().begin();i!=header.channels().end();++i)
    {
        std::string n(i.name()); 
        if(n=="ZBack")
        {
            _zback=true;
        }
        else if(n=="Z")
        {
            has_z=true;
        }
        else if(n=="A")
        {
            has_alpha=true;
        }
    }
    
    if(!has_z)
    {
        throw IEX_NAMESPACE::ArgExc("Deep data provided to CompositeDeepScanLine is missing a Z channel");
    }
    
    if(!has_alpha)
    {
        throw IEX_NAMESPACE::ArgExc("Deep data provided to CompositeDeepScanLine is missing an alpha channel");
    }
    
    
    if(_part.size()==0 && _file.size()==0)
    {
       // first in - update and return

       _dataWindow = header.dataWindow();
       
       return;
    }
    
    
    const Header * const match_header = _part.size()>0 ? &_part[0]->header() : &_file[0]->header();
    
    // check the sizes match
    if(match_header->displayWindow() != header.displayWindow())
    {
        throw IEX_NAMESPACE::ArgExc("Deep data provided to CompositeDeepScanLine has a different displayWindow to previously provided data");
    }
    
    _dataWindow.extendBy(header.dataWindow());
    
}
void 
CompositeDeepScanLine::Data::handleDeepFrameBuffer (DeepFrameBuffer& buf,
                                                    std::vector< unsigned int > & counts,
                                                    vector< std::vector< float* > > & pointers,
                                                    const Header& header,
                                                    int start,
                                                    int end)
{
    int width=_dataWindow.size().x+1;
    size_t pixelcount = width * (end-start+1);
    pointers.resize(_channels.size());
    counts.resize(pixelcount);
    buf.insertSampleCountSlice (Slice (OPENEXR_IMF_INTERNAL_NAMESPACE::UINT,
                                (char *) (&counts[0]-_dataWindow.min.x-start*width),
                                sizeof(unsigned int),
                                sizeof(unsigned int)*width));

    pointers[0].resize(pixelcount);
    buf.insert ("Z", DeepSlice (OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT,
                                (char *)(&pointers[0][0]-_dataWindow.min.x-start*width),
                                sizeof(float *),
                                sizeof(float *)*width,
                                sizeof(float) ));

    if(_zback)
    {
        pointers[1].resize(pixelcount);
        buf.insert ("ZBack", DeepSlice (OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT,
                                        (char *)(&pointers[1][0]-_dataWindow.min.x-start*width),
                                        sizeof(float *),
                                        sizeof(float *)*width,
                                        sizeof(float) ));
    }

    pointers[2].resize(pixelcount);
    buf.insert ("A", DeepSlice (OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT,
                                (char *)(&pointers[2][0]-_dataWindow.min.x-start*width),
                                sizeof(float *),
                                sizeof(float *)*width,
                                sizeof(float) ));


    size_t i =0;
    for(FrameBuffer::ConstIterator qt  = _outputFrameBuffer.begin();
                                   qt != _outputFrameBuffer.end();
                                   qt++)
    {
        int channel_in_source = _bufferMap[i];
        if(channel_in_source>2)
        {
            // not dealt with yet (0,1,2 previously inserted)
            pointers[channel_in_source].resize(pixelcount);
            buf.insert (qt.name(),
                        DeepSlice (OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT,
                                   (char *)(&pointers[channel_in_source][0]-_dataWindow.min.x-start*width),
                                   sizeof(float *),
                                   sizeof(float *)*width,
                                   sizeof(float) ));
        }

        i++;
    }

}

void
CompositeDeepScanLine::setCompositing(DeepCompositing* c)
{
  _Data->_comp=c;
}

const IMATH_NAMESPACE::Box2i& CompositeDeepScanLine::dataWindow() const
{
  return  _Data->_dataWindow;
}


void
CompositeDeepScanLine::setFrameBuffer(const FrameBuffer& fr)
{
    
    //
    // count channels; build map between channels in frame buffer
    // and channels in internal buffers
    //
    
    _Data->_channels.resize(3);
    _Data->_channels[0]="Z";
    _Data->_channels[1]=_Data->_zback ? "ZBack" : "Z";
    _Data->_channels[2]="A";
    _Data->_bufferMap.resize(0);
    
    for(FrameBuffer::ConstIterator q=fr.begin();q!=fr.end();q++)
    {
        string name(q.name());
        if(name=="ZBack")
        {
            _Data->_bufferMap.push_back(1);
        }else if(name=="Z")
        {
            _Data->_bufferMap.push_back(0);
        }else if(name=="A")
        {
            _Data->_bufferMap.push_back(2);
        }else{
            _Data->_bufferMap.push_back(_Data->_channels.size());
            _Data->_channels.push_back(name);
        }
    }
    
  _Data->_outputFrameBuffer=fr;
}

namespace 
{
    
class LineCompositeTask : public Task
{
  public:

    LineCompositeTask ( TaskGroup* group ,
                        CompositeDeepScanLine::Data * data,
                    int y,
                    int start,
                    vector<const char*>* names,
                    vector<vector< vector<float *> > >* pointers,
                    vector<unsigned int>* total_sizes,
                    vector<unsigned int>* num_sources
                  ) : Task(group) ,
                     _Data(data),
                     _y(y),
                     _start(start),
                     _names(names),
                     _pointers(pointers),
                     _total_sizes(total_sizes),
                     _num_sources(num_sources)
                     {}

    virtual ~LineCompositeTask () {}

    virtual void                execute ();
    CompositeDeepScanLine::Data*         _Data;
    int                                  _y;
    int                                  _start;
    vector<const char *>*                _names;
    vector<vector< vector<float *> > >*  _pointers;
    vector<unsigned int>*                _total_sizes;
    vector<unsigned int>*                _num_sources;

};


void
composite_line(int y,
               int start,
               CompositeDeepScanLine::Data * _Data,
               vector<const char *> & names,
               const vector<vector< vector<float *> > >  & pointers,
               const vector<unsigned int> & total_sizes,
               const vector<unsigned int> & num_sources
              )
{
    vector<float> output_pixel(names.size());    //the pixel we'll output to
    vector<const float *> inputs(names.size());
    DeepCompositing d; // fallback compositing engine
    DeepCompositing * comp= _Data->_comp ? _Data->_comp : &d;

    int pixel = (y-start)*(_Data->_dataWindow.max.x+1-_Data->_dataWindow.min.x);
    
     for(int x=_Data->_dataWindow.min.x;x<=_Data->_dataWindow.max.x;x++)
     {
           // set inputs[] to point to the first sample of the first part of each channel
           // if there's a zback, set all channel independently...

          if(_Data->_zback)
          {

              for(size_t channel=0;channel<names.size();channel++)
              {
                 inputs[channel]=pointers[0][channel][pixel];
              }

          }else{

              // otherwise, set 0 and 1 to point to Z


              inputs[0]=pointers[0][0][pixel];
              inputs[1]=pointers[0][0][pixel];
              for(size_t channel=2;channel<names.size();channel++)
              {
                  inputs[channel]=pointers[0][channel][pixel];
              }

          }
          comp->composite_pixel(&output_pixel[0],
                                &inputs[0],
                                &names[0],
                                names.size(),
                                total_sizes[pixel],
                                num_sources[pixel]
                               );


           size_t channel_number=0;


           //
           // write out composited value into internal frame buffer
           //
           for(FrameBuffer::Iterator it = _Data->_outputFrameBuffer.begin();it !=_Data->_outputFrameBuffer.end();it++)
           {

               float value = output_pixel[ _Data->_bufferMap[channel_number] ]; // value to write


                // cast to half float if necessary
               if(it.slice().type==OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT)
               {
                   * (float *)(it.slice().base + y*it.slice().yStride + x*it.slice().xStride) = value;
               }
               else if(it.slice().type==HALF)
               {
                   * (half *)(it.slice().base + y*it.slice().yStride + x*it.slice().xStride) = half(value);
               }

               channel_number++;

           }

           pixel++;

       }// next pixel on row
}

void LineCompositeTask::execute()
{
  composite_line(_y,_start,_Data,*_names,*_pointers,*_total_sizes,*_num_sources);
}


}

void
CompositeDeepScanLine::readPixels(int start, int end)
{
   size_t parts = _Data->_file.size() + _Data->_part.size(); // total of files+parts
   
   vector<DeepFrameBuffer> framebuffers(parts);
   vector< vector<unsigned int> > counts(parts);
   
   //
   // for each part, a pointer to an array of channels
   //
   vector<vector< vector<float *> > > pointers(parts);
   vector<const Header *> headers(parts);
   
   {
     size_t i;
     for(i=0;i<_Data->_file.size();i++)
     {
         headers[i] = &_Data->_file[i]->header();
     }
     
     for(size_t j=0;j<_Data->_part.size();j++)
     {
        headers[i+j] = &_Data->_part[j]->header();
     }
   }
   
   
   for(size_t i=0;i<parts;i++)
   {
     _Data->handleDeepFrameBuffer(framebuffers[i],counts[i],pointers[i],*headers[i],start,end);
   }
   
   //
   // set frame buffers and read scanlines from all parts
   // TODO what happens if SCANLINE not in data window?
   //
   
   {
       size_t i=0;
       for(i=0;i<_Data->_file.size();i++)
       {
            _Data->_file[i]->setFrameBuffer(framebuffers[i]);
            _Data->_file[i]->readPixelSampleCounts(start,end);
       }
       for(size_t j=0;j<_Data->_part.size();j++)
       {
           _Data->_part[j]->setFrameBuffer(framebuffers[i+j]);
           _Data->_part[j]->readPixelSampleCounts(start,end); 
       }
   }   
   
   
   //
   //  total width
   //
   
   size_t total_width = _Data->_dataWindow.size().x+1;
   size_t total_pixels = total_width*(end-start+1);
   vector<unsigned int> total_sizes(total_pixels);
   vector<unsigned int> num_sources(total_pixels); //number of parts with non-zero sample count
   
   size_t overall_sample_count=0; // sum of all samples in all images between start and end
   
   
   //
   // accumulate pixel counts
   //
   for(size_t ptr=0;ptr<total_pixels;ptr++)
   {
       total_sizes[ptr]=0;
       num_sources[ptr]=0;
       for(size_t j=0;j<parts;j++)
       {
          total_sizes[ptr]+=counts[j][ptr];
          if(counts[j][ptr]>0) num_sources[ptr]++;
       }
       overall_sample_count+=total_sizes[ptr];
       
       
       
   }
   
  
  
   
   //
   // allocate arrays for pixel data
   // samples array accessed as in pixels[channel][sample]
   //
   
   vector<vector<float> > samples( _Data->_channels.size() );
   
   for(size_t channel=0;channel<_Data->_channels.size();channel++)
   {
       if( channel!=1 || _Data->_zback)
       {            
           samples[channel].resize(overall_sample_count);
       }
   }
   
   for(size_t channel=0;channel<samples.size();channel++)
   {
       
       if( channel!=1 || _Data->_zback)
       {
           
           samples[channel].resize(overall_sample_count);
       
       
          //
          // allocate pointers for channel data
          //
          
          size_t offset=0;
       
          for(size_t pixel=0;pixel<total_pixels;pixel++)
          {
              for(size_t part=0 ; part<parts && offset<overall_sample_count ; part++ )
              {
                      pointers[part][channel][pixel]=&samples[channel][offset];           
                      offset+=counts[part][pixel];
              }
          }
       
       }
   }
   
   //
   // read data
   //
   
   for(size_t i=0;i<_Data->_file.size();i++)
   {
       _Data->_file[i]->readPixels(start,end);
   }
   for(size_t j=0;j<_Data->_part.size();j++)
   {
       _Data->_part[j]->readPixels(start,end); 
   }
   
   
   
   
   //
   // composite pixels and write back to framebuffer
  //
   
   
   // turn vector of strings into array of char *
   // and make sure 'ZBack' channel is correct
   vector<const char *> names(_Data->_channels.size());
   for(size_t i=0;i<names.size();i++)
   {
       names[i]=_Data->_channels[i].c_str();
   }
   
   if(!_Data->_zback) names[1]=names[0]; // no zback channel, so make it point to z

   
   
   TaskGroup g;
   for(int y=start;y<=end;y++)
   {
       ThreadPool::addGlobalTask(new LineCompositeTask(&g,_Data,y,start,&names,&pointers,&total_sizes,&num_sources));
   }//next row
}  

const FrameBuffer& 
CompositeDeepScanLine::frameBuffer() const
{
  return _Data->_outputFrameBuffer;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
