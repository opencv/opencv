///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2009-2014 DreamWorks Animation LLC. 
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
// *       Neither the name of DreamWorks Animation nor the names of
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

//
// A program to generate various acceleration lookup tables 
// for Imf::DwaCompressor
//

#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include <OpenEXRConfig.h>

#ifdef OPENEXR_IMF_HAVE_SYSCONF_NPROCESSORS_ONLN
#include <unistd.h>
#endif

#include <half.h>
#include <IlmThread.h>
#include <IlmThreadSemaphore.h>
#include <ImfIO.h>
#include <ImfXdr.h>
#include "ImfNamespace.h"

using namespace OPENEXR_IMF_NAMESPACE;

namespace {

    class LutHeaderWorker
    {
        public:
            class Runner : public ILMTHREAD_NAMESPACE::Thread
            {
                public:
                    Runner(LutHeaderWorker &worker, bool output):
                        ILMTHREAD_NAMESPACE::Thread(),
                        _worker(worker),
                        _output(output)
                    {
                        start();
                    }

                    virtual ~Runner()
                    {
                        _semaphore.wait();
                    }

                    virtual void run()
                    {
                        _semaphore.post();
                        _worker.run(_output);
                    }

                private:
                    LutHeaderWorker     &_worker;
                    bool                 _output;
                    ILMTHREAD_NAMESPACE::Semaphore _semaphore;

            }; // class LutHeaderWorker::Runner


            LutHeaderWorker(size_t startValue,
                            size_t endValue):
                _lastCandidateCount(0),
                _startValue(startValue),
                _endValue(endValue),
                _numElements(0),
                _offset(new size_t[numValues()]),
                _elements(new unsigned short[1024*1024*2])
            {
            }

            ~LutHeaderWorker()
            {
                delete[] _offset;
                delete[] _elements;
            }

            size_t lastCandidateCount() const
            {
                return _lastCandidateCount;
            }

            size_t numValues() const 
            {
                return _endValue - _startValue;
            }

            size_t numElements() const
            {
                return _numElements;
            }

            const size_t* offset() const
            {
                return _offset;
            }

            const unsigned short* elements() const
            {
                return _elements;
            }

            void run(bool outputProgress)
            {
                half candidate[16];
                int  candidateCount = 0;

                for (size_t input=_startValue; input<_endValue; ++input) {

                    if (outputProgress) {
#ifdef __GNUC__
                        if (input % 100 == 0) {
                            fprintf(stderr, 
                            " Building acceleration for DwaCompressor, %.2f %%      %c",
                                          100.*(float)input/(float)numValues(), 13);
                        }
#else
                        if (input % 1000 == 0) {
                            fprintf(stderr, 
                            " Building acceleration for DwaCompressor, %.2f %%\n",
                                          100.*(float)input/(float)numValues());
                        }
#endif
                    } 

                    
                    int  numSetBits = countSetBits(input);
                    half inputHalf, closestHalf;

                    inputHalf.setBits(input);

                    _offset[input - _startValue] = _numElements;

                    // Gather candidates
                    candidateCount = 0;
                    for (int targetNumSetBits=numSetBits-1; targetNumSetBits>=0;
                                                           --targetNumSetBits) {
                        bool valueFound = false;

                        for (int i=0; i<65536; ++i) {
                            if (countSetBits(i) != targetNumSetBits) continue;

                            if (!valueFound) {
                                closestHalf.setBits(i);
                                valueFound = true;
                            } else {
                                half tmpHalf;

                                tmpHalf.setBits(i);

                                if (fabs((float)inputHalf - (float)tmpHalf) < 
                                    fabs((float)inputHalf - (float)closestHalf)) {
                                    closestHalf = tmpHalf;
                                }
                            }
                        }

                        if (valueFound == false) {
                            fprintf(stderr, "bork bork bork!\n");
                        }       

                        candidate[candidateCount] = closestHalf;
                        candidateCount++;
                    }

                    // Sort candidates by increasing number of bits set
                    for (int i=0; i<candidateCount; ++i) {
                        for (int j=i+1; j<candidateCount; ++j) {

                            int   iCnt = countSetBits(candidate[i].bits());
                            int   jCnt = countSetBits(candidate[j].bits());

                            if (jCnt < iCnt) {
                                half tmp     = candidate[i];
                                candidate[i] = candidate[j];
                                candidate[j] = tmp;
                            }
                        }
                    }

                    // Copy candidates to the data buffer;
                    for (int i=0; i<candidateCount; ++i) {
                        _elements[_numElements] = candidate[i].bits();
                        _numElements++;
                    }

                    if (input == _endValue-1) {
                        _lastCandidateCount = candidateCount;
                    }
                }
            }
            

        private:
            size_t          _lastCandidateCount;
            size_t          _startValue;
            size_t          _endValue;
            size_t          _numElements;
            size_t         *_offset;
            unsigned short *_elements;

            //
            // Precomputing the bit count runs faster than using
            // the builtin instruction, at least in one case..
            //
            // Precomputing 8-bits is no slower than 16-bits,
            // and saves a fair bit of overhead..
            //
            int countSetBits(unsigned short src)
            {
                static const unsigned short numBitsSet[256] =
                {
                    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
                };

                return numBitsSet[src & 0xff] + numBitsSet[src >> 8];
            }

    }; // class LutHeaderWorker

} // namespace


//
// Generate a no-op LUT, to cut down in conditional branches
//
void
generateNoop()
{
    printf("const unsigned short dwaCompressorNoOp[] = \n");
    printf("{");
    for (int i=0; i<65536; ++i) {

        if (i % 8 == 0) {
            printf("\n    ");
        }

        unsigned short dst;
        char *tmp = (char *)(&dst);

        unsigned short src = (unsigned short)i;
        Xdr::write <CharPtrIO> (tmp,  src);

        printf("0x%04x, ", dst);
    }
    printf("\n};\n");
}

//
// Nonlinearly encode luminance. For values below 1.0, we want
// to use a gamma 2.2 function to match what is fairly common
// for storing output referred. However, > 1, gamma functions blow up,
// and log functions are much better behaved. We could use a log 
// function everywhere, but it tends to over-sample dark 
// regions and undersample the brighter regions, when 
// compared to the way real devices reproduce values.
//
// So, above 1, use a log function which is a smooth blend
// into the gamma function. 
//
//  Nonlinear(linear) = 
//
//    linear^(1./2.2)             / linear <= 1.0
//                               |
//    ln(linear)/ln(e^2.2) + 1    \ otherwise
//
//
// toNonlinear[] needs to take in XDR format half float values,
// and output NATIVE format float. 
//
// toLinear[] does the opposite - takes in NATIVE half and 
// outputs XDR half values.
//

void
generateToLinear()
{
    unsigned short toLinear[65536];

    toLinear[0] = 0;

    for (int i=1; i<65536; ++i) {
        half  h;
        float sign    = 1;
        float logBase = pow(2.7182818, 2.2);

        // map  NaN and inf to 0
        if ((i & 0x7c00) == 0x7c00) {
            toLinear[i]    = 0;
            continue;
        }

        //
        // _toLinear - assume i is NATIVE, but our output needs
        //             to get flipped to XDR
        //
        h.setBits(i);
        sign = 1;
        if ((float)h < 0) {
            sign = -1;
        } 

        if ( fabs( (float)h) <= 1.0 ) {
            h  = (half)(sign * pow((float)fabs((float)h), 2.2f));
        } else {
            h  = (half)(sign * pow(logBase, (float)(fabs((float)h) - 1.0)));
        }

        {
            char *tmp = (char *)(&toLinear[i]);

            Xdr::write <CharPtrIO> ( tmp,  h.bits());
        }
    }
    
    printf("const unsigned short dwaCompressorToLinear[] = \n");
    printf("{");
    for (int i=0; i<65536; ++i) {
        if (i % 8 == 0) {
            printf("\n    ");
        }
        printf("0x%04x, ", toLinear[i]);
    }
    printf("\n};\n");
}


void
generateToNonlinear()
{
    unsigned short toNonlinear[65536];

    toNonlinear[0] = 0;

    for (int i=1; i<65536; ++i) {
        unsigned short usNative, usXdr;
        half  h;
        float sign    = 1;
        float logBase = pow(2.7182818, 2.2);

        usXdr           = i;

        {
            const char *tmp = (char *)(&usXdr);

            Xdr::read<CharPtrIO>(tmp, usNative);
        }

        // map  NaN and inf to 0
        if ((usNative & 0x7c00) == 0x7c00) {
            toNonlinear[i] = 0;
            continue;
        }

        //
        // toNonlinear - assume i is XDR
        //
        h.setBits(usNative);
        sign = 1;
        if ((float)h < 0) {
            sign = -1;
        } 

        if ( fabs( (float)h ) <= 1.0) {
            h = (half)(sign * pow(fabs((float)h), 1.f/2.2f));
        } else {
            h = (half)(sign * ( log(fabs((float)h)) / log(logBase) + 1.0) );
        }
        toNonlinear[i] = h.bits();
    }

    printf("const unsigned short dwaCompressorToNonlinear[] = \n");
    printf("{");
    for (int i=0; i<65536; ++i) {
        if (i % 8 == 0) {
            printf("\n    ");
        }
        printf("0x%04x, ", toNonlinear[i]);
    }
    printf("\n};\n");
}

//
// Attempt to get available CPUs in a somewhat portable way. 
//

int
cpuCount()
{
    if (!ILMTHREAD_NAMESPACE::supportsThreads()) return 1;

    int cpuCount = 1;

#if defined (OPENEXR_IMF_HAVE_SYSCONF_NPROCESSORS_ONLN)

    cpuCount = sysconf(_SC_NPROCESSORS_ONLN);

#elif defined (_WIN32)

    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );
    cpuCount = sysinfo.dwNumberOfProcessors;

#endif

    if (cpuCount < 1) cpuCount = 1;
    return cpuCount;
}

//
// Generate acceleration luts for the quantization.
//
// For each possible input value, we want to find the closest numbers
// which have one fewer bits set than before. 
//
// This gives us num_bits(input)-1 values per input. If we alloc
// space for everything, that's like a 2MB table. We can do better
// by compressing all the values to be contigious and using offset
// pointers.
//
// After we've found the candidates with fewer bits set, sort them
// based on increasing numbers of bits set. This way, on quantize(),
// we can scan through the list and halt once we find the first
// candidate within the error range. For small values that can 
// be quantized to 0, 0 is the first value tested and the search
// can exit fairly quickly.
//

void
generateLutHeader()
{
    std::vector<LutHeaderWorker*> workers;

    size_t numWorkers     = cpuCount();
    size_t workerInterval = 65536 / numWorkers;

    for (size_t i=0; i<numWorkers; ++i) {
        if (i != numWorkers-1) {
            workers.push_back( new LutHeaderWorker( i   *workerInterval, 
                                                   (i+1)*workerInterval) );
        } else {
            workers.push_back( new LutHeaderWorker(i*workerInterval, 65536) );
        }
    }

    if (ILMTHREAD_NAMESPACE::supportsThreads()) {
        std::vector<LutHeaderWorker::Runner*> runners;
        for (size_t i=0; i<workers.size(); ++i) {
            runners.push_back( new LutHeaderWorker::Runner(*workers[i], (i==0)) );
        }

        for (size_t i=0; i<workers.size(); ++i) {
            delete runners[i];
        }
    } else {
        for (size_t i=0; i<workers.size(); ++i) {
            workers[i]->run(i == 0);
        }
    }

    printf("static unsigned int closestDataOffset[] = {\n");
    int offsetIdx  = 0;
    int offsetPrev = 0;
    for (size_t i=0; i<workers.size(); ++i) {
        for (size_t value=0; value<workers[i]->numValues(); ++value) {
            if (offsetIdx % 8 == 0) {
                printf("    ");
            }
            printf("%6lu, ", workers[i]->offset()[value] + offsetPrev);
            if (offsetIdx % 8 == 7) {
                printf("\n");
            }
            offsetIdx++;
        }
        offsetPrev += workers[i]->offset()[workers[i]->numValues()-1] + 
                      workers[i]->lastCandidateCount();
    }
    printf("};\n\n\n");


    printf("static unsigned short closestData[] = {\n");
    int elementIdx = 0;
    for (size_t i=0; i<workers.size(); ++i) {
        for (size_t element=0; element<workers[i]->numElements(); ++element) {
            if (elementIdx % 8 == 0) {
                printf("    ");
            }
            printf("%5d, ", workers[i]->elements()[element]);
            if (elementIdx % 8 == 7) {
                printf("\n");
            }
            elementIdx++;
        }    
    }
    printf("};\n\n\n");

    for (size_t i=0; i<workers.size(); ++i) {
        delete workers[i];
    }
}


int
main(int argc, char **argv)
{
    printf("#include <cstddef>\n");
    printf("\n\n\n");

    generateNoop();

    printf("\n\n\n");

    generateToLinear();

    printf("\n\n\n");

    generateToNonlinear();

    printf("\n\n\n");

    generateLutHeader();

    return 0;
}
