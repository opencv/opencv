/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSegmentedPaletteColorLookupTable.h"
#include "gdcmDataSet.h"

// http://blog.goo.ne.jp/satomi_takeo/e/3643e5249b2a9650f9e10ef1c830e8b8
//
#include <map>
#include <algorithm>
#include <iterator>
#include <vector>
#include <deque>

namespace gdcm
{
    // abstract class for segment.
    template <typename EntryType>
    class Segment {
    public:
        virtual ~Segment() {}
        typedef std::map<const EntryType*, const Segment*> SegmentMap;
        virtual bool Expand(const SegmentMap& instances,
            std::vector<EntryType>& expanded) const = 0;
        const EntryType* First() const { return _first; }
        const EntryType* Last() const { return _last; }
        struct ToMap {
            std::pair<
                typename SegmentMap::key_type,
                typename SegmentMap::mapped_type
            >
                operator()(const Segment* segment) const
            { return std::make_pair(segment->First(), segment); }
        };
    protected:
        Segment(const EntryType* first, const EntryType* last) {
            _first = first; _last = last;
        }
        const EntryType* _first;
        const EntryType* _last;
    };

    // discrete segment (opcode = 0)
    template <typename EntryType>
    class DiscreteSegment : public Segment<EntryType> {
    public:
        typedef typename Segment<EntryType>::SegmentMap SegmentMap;
        DiscreteSegment(const EntryType* first)
            : Segment<EntryType>(first, first+2+*(first+1)) {}
        virtual bool Expand(const SegmentMap&,
            std::vector<EntryType>& expanded) const
        {
            std::copy(this->_first + 2, this->_last, std::back_inserter(expanded));
            return true;
        }
    };

    // linear segment (opcode = 1)
    template <typename EntryType>
    class LinearSegment : public Segment<EntryType> {
    public:
        typedef typename Segment<EntryType>::SegmentMap SegmentMap;
        LinearSegment(const EntryType* first)
            : Segment<EntryType>(first, first+3) {}
        virtual bool Expand(const SegmentMap&,
            std::vector<EntryType>& expanded) const
        {
            if ( expanded.empty() ) {
                // linear segment can't be the first segment.
                return false;
            }
            EntryType length = *(this->_first + 1);
            EntryType y0 = expanded.back();
            EntryType y1 = *(this->_first + 2);
            double y01 = y1 - y0;
            for ( EntryType i = 0; i <length; ++i ) {
                double value_float
                    = static_cast<double>(y0)
                    + (static_cast<double>(i)/static_cast<double>(length)) * y01;
                EntryType value_int = static_cast<EntryType>(value_float + 0.5);
                expanded.push_back(value_int);
            }
            return true;
        }
    };

    // indirect segment (opcode = 2)
    template <typename EntryType>
    class IndirectSegment : public Segment<EntryType> {
    public:
        typedef typename Segment<EntryType>::SegmentMap SegmentMap;
        IndirectSegment(const EntryType* first)
            : Segment<EntryType>(first, first+2+4/sizeof(EntryType)) {}
        virtual bool Expand(const SegmentMap& instances,
            std::vector<EntryType>& expanded) const
        {
            if ( instances.empty() ) {
                // some other segments are required as references.
                return false;
            }
            const EntryType* first_segment = instances.begin()->first;
            const unsigned short* pOffset
                = reinterpret_cast<const unsigned short*>(this->_first + 2);
            unsigned long offsetBytes
                = (*pOffset) | (static_cast<unsigned long>(*(pOffset + 1)) << 16);
            const EntryType* copied_part_head
                = first_segment + offsetBytes / sizeof(EntryType);
            typename SegmentMap::const_iterator ppHeadSeg = instances.find(copied_part_head);
            if ( ppHeadSeg == instances.end() ) {
                // referred segment not found
                return false;
            }
            EntryType nNumCopies = *(this->_first + 1);
            typename SegmentMap::const_iterator ppSeg = ppHeadSeg;
            while ( std::distance(ppHeadSeg, ppSeg) <nNumCopies ) {
                assert( ppSeg != instances.end() );
                ppSeg->second->Expand(instances, expanded);
                ++ppSeg;
            }
            return true;
        }
    };

    template <typename EntryType>
    void ExpandPalette(const EntryType* raw_values, uint32_t length,
        std::vector<EntryType>& palette)
    {
        typedef std::deque<Segment<EntryType>*> SegmentList;
        SegmentList segments;
        const EntryType* raw_seg = raw_values;
        while ( (std::distance(raw_values, raw_seg) * sizeof(EntryType)) <length ) {
            Segment<EntryType>* segment = NULL;
            if ( *raw_seg == 0 ) {
                segment = new DiscreteSegment<EntryType>(raw_seg);
            } else if ( *raw_seg == 1 ) {
                segment = new LinearSegment<EntryType>(raw_seg);
            } else if ( *raw_seg == 2 ) {
                segment = new IndirectSegment<EntryType>(raw_seg);
            }
            if ( segment ) {
                segments.push_back(segment);
                raw_seg = segment->Last();
            } else {
                // invalid opcode
                break;
            }
        }
        typename Segment<EntryType>::SegmentMap instances;
        std::transform(segments.begin(), segments.end(),
            std::inserter(instances, instances.end()), typename Segment<EntryType>::ToMap());
        typename SegmentList::iterator ppSeg = segments.begin();
        typename SegmentList::iterator endOfSegments = segments.end();
        for ( ; ppSeg != endOfSegments; ++ppSeg ) {
            (*ppSeg)->Expand(instances, palette);
        }
        ppSeg = segments.begin();
        for ( ; ppSeg != endOfSegments; ++ppSeg ) {
            delete *ppSeg;
        }
    }

SegmentedPaletteColorLookupTable::SegmentedPaletteColorLookupTable()
{
}

SegmentedPaletteColorLookupTable::~SegmentedPaletteColorLookupTable()
{
}

void SegmentedPaletteColorLookupTable::SetLUT(LookupTableType type, const unsigned char *array,
    unsigned int length)
{
  if( BitSample == 8 )
    {
    assert(0); // TODO
    }
  else if( BitSample == 16 )
    {
    const uint16_t *array16 = (uint16_t*)array;
    const uint16_t *segment_values = array16;
    std::vector<uint16_t> palette;
    unsigned int num_entries = GetLUTLength(type);
    palette.reserve(num_entries);
    assert( length % 2 == 0 );
    // FIXME: inplace byteswapping (BAD!)
    SwapperNoOp::SwapArray((uint16_t*)segment_values,length/2);
    ExpandPalette(segment_values, length, palette);
    LookupTable::SetLUT(type, (unsigned char*)&palette[0], (unsigned int)(palette.size() * 2));
    }
}

} // end namespace gdcm
