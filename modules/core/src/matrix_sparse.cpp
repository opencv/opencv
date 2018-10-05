// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "opencv2/core/mat.hpp"
#include "opencv2/core/types_c.h"
#include "precomp.hpp"

namespace cv {

template<typename T1, typename T2> void
convertData_(const void* _from, void* _to, int cn)
{
    const T1* from = (const T1*)_from;
    T2* to = (T2*)_to;
    if( cn == 1 )
        *to = saturate_cast<T2>(*from);
    else
        for( int i = 0; i < cn; i++ )
            to[i] = saturate_cast<T2>(from[i]);
}

template<typename T1, typename T2> void
convertScaleData_(const void* _from, void* _to, int cn, double alpha, double beta)
{
    const T1* from = (const T1*)_from;
    T2* to = (T2*)_to;
    if( cn == 1 )
        *to = saturate_cast<T2>(*from*alpha + beta);
    else
        for( int i = 0; i < cn; i++ )
            to[i] = saturate_cast<T2>(from[i]*alpha + beta);
}

typedef void (*ConvertData)(const void* from, void* to, int cn);
typedef void (*ConvertScaleData)(const void* from, void* to, int cn, double alpha, double beta);

static ConvertData getConvertElem(int fromType, int toType)
{
    static const std::map<int, std::map<int, ConvertData> > tab
    {
        {CV_8U, {
            {CV_8U,  convertData_<uchar    , uchar    >},
            {CV_16U, convertData_<uchar    , ushort   >},
            {CV_32U, convertData_<uchar    , uint     >},
            {CV_64U, convertData_<uchar    , uint64_t >},
            {CV_8S,  convertData_<uchar    , schar    >},
            {CV_16S, convertData_<uchar    , short    >},
            {CV_32S, convertData_<uchar    , int      >},
            {CV_64S, convertData_<uchar    , int64_t  >},
            //{CV_16F, convertData_<uchar    , float16_t>},
            {CV_32F, convertData_<uchar    , float    >},
            {CV_64F, convertData_<uchar    , double   >},
        }},
        {CV_16U, {
            {CV_8U,  convertData_<ushort   , uchar    >},
            {CV_16U, convertData_<ushort   , ushort   >},
            {CV_32U, convertData_<ushort   , uint     >},
            {CV_64U, convertData_<ushort   , uint64_t >},
            {CV_8S,  convertData_<ushort   , schar    >},
            {CV_16S, convertData_<ushort   , short    >},
            {CV_32S, convertData_<ushort   , int      >},
            {CV_64S, convertData_<ushort   , int64_t  >},
            //{CV_16F, convertData_<ushort   , float16_t>},
            {CV_32F, convertData_<ushort   , float    >},
            {CV_64F, convertData_<ushort   , double   >},
        }},
        {CV_32U, {
            {CV_8U,  convertData_<uint     , uchar    >},
            {CV_16U, convertData_<uint     , ushort   >},
            {CV_32U, convertData_<uint     , uint     >},
            {CV_64U, convertData_<uint     , uint64_t >},
            {CV_8S,  convertData_<uint     , schar    >},
            {CV_16S, convertData_<uint     , short    >},
            {CV_32S, convertData_<uint     , int      >},
            {CV_64S, convertData_<uint     , int64_t  >},
            //{CV_16F, convertData_<uint     , float16_t>},
            {CV_32F, convertData_<uint     , float    >},
            {CV_64F, convertData_<uint     , double   >},
        }},
        {CV_64U, {
            {CV_8U,  convertData_<uint64_t , uchar    >},
            {CV_16U, convertData_<uint64_t , ushort   >},
            {CV_32U, convertData_<uint64_t , uint     >},
            {CV_64U, convertData_<uint64_t , uint64_t >},
            {CV_8S,  convertData_<uint64_t , schar    >},
            {CV_16S, convertData_<uint64_t , short    >},
            {CV_32S, convertData_<uint64_t , int      >},
            {CV_64S, convertData_<uint64_t , int64_t  >},
            //{CV_16F, convertData_<uint64_t , float16_t>},
            {CV_32F, convertData_<uint64_t , float    >},
            {CV_64F, convertData_<uint64_t , double   >},
        }},
        {CV_8S, {
            {CV_8U,  convertData_<schar    , uchar    >},
            {CV_16U, convertData_<schar    , ushort   >},
            {CV_32U, convertData_<schar    , uint     >},
            {CV_64U, convertData_<schar    , uint64_t >},
            {CV_8S,  convertData_<uchar    , schar    >},
            {CV_16S, convertData_<schar    , short    >},
            {CV_32S, convertData_<schar    , int      >},
            {CV_64S, convertData_<schar    , int64_t  >},
            //{CV_16F, convertData_<schar    , float16_t>},
            {CV_32F, convertData_<schar    , float    >},
            {CV_64F, convertData_<schar    , double   >},
        }},
        {CV_16S, {
            {CV_8U,  convertData_<short    , uchar    >},
            {CV_16U, convertData_<short    , ushort   >},
            {CV_32U, convertData_<short    , uint     >},
            {CV_64U, convertData_<short    , uint64_t >},
            {CV_8S,  convertData_<short    , schar    >},
            {CV_16S, convertData_<short    , short    >},
            {CV_32S, convertData_<short    , int      >},
            {CV_64S, convertData_<short    , int64_t  >},
            //{CV_16F, convertData_<short    , float16_t>},
            {CV_32F, convertData_<short    , float    >},
            {CV_64F, convertData_<short    , double   >},
        }},
        {CV_32S, {
            {CV_8U,  convertData_<int      , uchar    >},
            {CV_16U, convertData_<int      , ushort   >},
            {CV_32U, convertData_<int      , uint     >},
            {CV_64U, convertData_<int      , uint64_t >},
            {CV_8S,  convertData_<int      , schar    >},
            {CV_16S, convertData_<int      , short    >},
            {CV_32S, convertData_<int      , int      >},
            {CV_64S, convertData_<int      , int64_t  >},
            //{CV_16F, convertData_<int      , float16_t>},
            {CV_32F, convertData_<int      , float    >},
            {CV_64F, convertData_<int      , double   >},
        }},
        {CV_64S, {
            {CV_8U,  convertData_<int64_t  , uchar    >},
            {CV_16U, convertData_<int64_t  , ushort   >},
            {CV_32U, convertData_<int64_t  , uint     >},
            {CV_64U, convertData_<int64_t  , uint64_t >},
            {CV_8S,  convertData_<int64_t  , schar    >},
            {CV_16S, convertData_<int64_t  , short    >},
            {CV_32S, convertData_<int64_t  , int      >},
            {CV_64S, convertData_<int64_t  , int64_t  >},
            //{CV_16F, convertData_<int64_t  , float16_t>},
            {CV_32F, convertData_<int64_t  , float    >},
            {CV_64F, convertData_<int64_t  , double   >},
        }},
        /*{CV_16F, {
            {CV_8U,  convertData_<float16_t, uchar    >},
            {CV_16U, convertData_<float16_t, ushort   >},
            {CV_32U, convertData_<float16_t, uint     >},
            {CV_64U, convertData_<float16_t, uint64_t >},
            {CV_8S,  convertData_<float16_t, schar    >},
            {CV_16S, convertData_<float16_t, short    >},
            {CV_32S, convertData_<float16_t, int      >},
            {CV_64S, convertData_<float16_t, int64_t  >},
            {CV_16F, convertData_<float16_t, float16_t>},
            {CV_32F, convertData_<float16_t, float    >},
            {CV_64F, convertData_<float16_t, double   >},
        }},*/
        {CV_32F, {
            {CV_8U,  convertData_<float    , uchar    >},
            {CV_16U, convertData_<float    , ushort   >},
            {CV_32U, convertData_<float    , uint     >},
            {CV_64U, convertData_<float    , uint64_t >},
            {CV_8S,  convertData_<float    , schar    >},
            {CV_16S, convertData_<float    , short    >},
            {CV_32S, convertData_<float    , int      >},
            {CV_64S, convertData_<float    , int64_t  >},
            //{CV_16F, convertData_<float    , float16_t>},
            {CV_32F, convertData_<float    , float    >},
            {CV_64F, convertData_<float    , double   >},
        }},
        {CV_64F, {
            {CV_8U,  convertData_<double   , uchar    >},
            {CV_16U, convertData_<double   , ushort   >},
            {CV_32U, convertData_<double   , uint     >},
            {CV_64U, convertData_<double   , uint64_t >},
            {CV_8S,  convertData_<double   , schar    >},
            {CV_16S, convertData_<double   , short    >},
            {CV_32S, convertData_<double   , int      >},
            {CV_64S, convertData_<double   , int64_t  >},
            //{CV_16F, convertData_<double   , float16_t>},
            {CV_32F, convertData_<double   , float    >},
            {CV_64F, convertData_<double   , double   >},
        }},
    };

    ConvertData func = tab.at(CV_MAT_DEPTH(fromType)).at(CV_MAT_DEPTH(toType));
    CV_Assert( func != NULL );
    return func;
}

static ConvertScaleData getConvertScaleElem(int fromType, int toType)
{
    static const std::map<int, std::map<int, ConvertScaleData> > map
    {
        {CV_8U, {
            {CV_8U,  convertScaleData_<uchar    , uchar    >},
            {CV_16U, convertScaleData_<uchar    , ushort   >},
            {CV_32U, convertScaleData_<uchar    , uint     >},
            {CV_64U, convertScaleData_<uchar    , uint64_t >},
            {CV_8S,  convertScaleData_<uchar    , schar    >},
            {CV_16S, convertScaleData_<uchar    , short    >},
            {CV_32S, convertScaleData_<uchar    , int      >},
            {CV_64S, convertScaleData_<uchar    , int64_t  >},
            //{CV_16F, convertScaleData_<uchar    , float16_t>},
            {CV_32F, convertScaleData_<uchar    , float    >},
            {CV_64F, convertScaleData_<uchar    , double   >},
        }},
        {CV_16U, {
            {CV_8U,  convertScaleData_<ushort   , uchar    >},
            {CV_16U, convertScaleData_<ushort   , ushort   >},
            {CV_32U, convertScaleData_<ushort   , uint     >},
            {CV_64U, convertScaleData_<ushort   , uint64_t >},
            {CV_8S,  convertScaleData_<ushort   , schar    >},
            {CV_16S, convertScaleData_<ushort   , short    >},
            {CV_32S, convertScaleData_<ushort   , int      >},
            {CV_64S, convertScaleData_<ushort   , int64_t  >},
            //{CV_16F, convertScaleData_<ushort   , float16_t>},
            {CV_32F, convertScaleData_<ushort   , float    >},
            {CV_64F, convertScaleData_<ushort   , double   >},
        }},
        {CV_32U, {
            {CV_8U,  convertScaleData_<uint     , uchar    >},
            {CV_16U, convertScaleData_<uint     , ushort   >},
            {CV_32U, convertScaleData_<uint     , uint     >},
            {CV_64U, convertScaleData_<uint     , uint64_t >},
            {CV_8S,  convertScaleData_<uint     , schar    >},
            {CV_16S, convertScaleData_<uint     , short    >},
            {CV_32S, convertScaleData_<uint     , int      >},
            {CV_64S, convertScaleData_<uint     , int64_t  >},
            //{CV_16F, convertScaleData_<uint     , float16_t>},
            {CV_32F, convertScaleData_<uint     , float    >},
            {CV_64F, convertScaleData_<uint     , double   >},
        }},
        {CV_64U, {
            {CV_8U,  convertScaleData_<uint64_t , uchar    >},
            {CV_16U, convertScaleData_<uint64_t , ushort   >},
            {CV_32U, convertScaleData_<uint64_t , uint     >},
            {CV_64U, convertScaleData_<uint64_t , uint64_t >},
            {CV_8S,  convertScaleData_<uint64_t , schar    >},
            {CV_16S, convertScaleData_<uint64_t , short    >},
            {CV_32S, convertScaleData_<uint64_t , int      >},
            {CV_64S, convertScaleData_<uint64_t , int64_t  >},
            //{CV_16F, convertScaleData_<uint64_t , float16_t>},
            {CV_32F, convertScaleData_<uint64_t , float    >},
            {CV_64F, convertScaleData_<uint64_t , double   >},
        }},
        {CV_8S, {
            {CV_8U,  convertScaleData_<schar    , uchar    >},
            {CV_16U, convertScaleData_<schar    , ushort   >},
            {CV_32U, convertScaleData_<schar    , uint     >},
            {CV_64U, convertScaleData_<schar    , uint64_t >},
            {CV_8S,  convertScaleData_<uchar    , schar    >},
            {CV_16S, convertScaleData_<schar    , short    >},
            {CV_32S, convertScaleData_<schar    , int      >},
            {CV_64S, convertScaleData_<schar    , int64_t  >},
            //{CV_16F, convertScaleData_<schar    , float16_t>},
            {CV_32F, convertScaleData_<schar    , float    >},
            {CV_64F, convertScaleData_<schar    , double   >},
        }},
        {CV_16S, {
            {CV_8U,  convertScaleData_<short    , uchar    >},
            {CV_16U, convertScaleData_<short    , ushort   >},
            {CV_32U, convertScaleData_<short    , uint     >},
            {CV_64U, convertScaleData_<short    , uint64_t >},
            {CV_8S,  convertScaleData_<short    , schar    >},
            {CV_16S, convertScaleData_<short    , short    >},
            {CV_32S, convertScaleData_<short    , int      >},
            {CV_64S, convertScaleData_<short    , int64_t  >},
            //{CV_16F, convertScaleData_<short    , float16_t>},
            {CV_32F, convertScaleData_<short    , float    >},
            {CV_64F, convertScaleData_<short    , double   >},
        }},
        {CV_32S, {
            {CV_8U,  convertScaleData_<int      , uchar    >},
            {CV_16U, convertScaleData_<int      , ushort   >},
            {CV_32U, convertScaleData_<int      , uint     >},
            {CV_64U, convertScaleData_<int      , uint64_t >},
            {CV_8S,  convertScaleData_<int      , schar    >},
            {CV_16S, convertScaleData_<int      , short    >},
            {CV_32S, convertScaleData_<int      , int      >},
            {CV_64S, convertScaleData_<int      , int64_t  >},
            //{CV_16F, convertScaleData_<int      , float16_t>},
            {CV_32F, convertScaleData_<int      , float    >},
            {CV_64F, convertScaleData_<int      , double   >},
        }},
        {CV_64S, {
            {CV_8U,  convertScaleData_<int64_t  , uchar    >},
            {CV_16U, convertScaleData_<int64_t  , ushort   >},
            {CV_32U, convertScaleData_<int64_t  , uint     >},
            {CV_64U, convertScaleData_<int64_t  , uint64_t >},
            {CV_8S,  convertScaleData_<int64_t  , schar    >},
            {CV_16S, convertScaleData_<int64_t  , short    >},
            {CV_32S, convertScaleData_<int64_t  , int      >},
            {CV_64S, convertScaleData_<int64_t  , int64_t  >},
            //{CV_16F, convertScaleData_<int64_t  , float16_t>},
            {CV_32F, convertScaleData_<int64_t  , float    >},
            {CV_64F, convertScaleData_<int64_t  , double   >},
        }},
        /*{CV_16F, {
            {CV_8U,  convertScaleData_<float16_t, uchar    >},
            {CV_16U, convertScaleData_<float16_t, ushort   >},
            {CV_32U, convertScaleData_<float16_t, uint     >},
            {CV_64U, convertScaleData_<float16_t, uint64_t >},
            {CV_8S,  convertScaleData_<float16_t, schar    >},
            {CV_16S, convertScaleData_<float16_t, short    >},
            {CV_32S, convertScaleData_<float16_t, int      >},
            {CV_64S, convertScaleData_<float16_t, int64_t  >},
            {CV_16F, convertScaleData_<float16_t, float16_t>},
            {CV_32F, convertScaleData_<float16_t, float    >},
            {CV_64F, convertScaleData_<float16_t, double   >},
        }},*/
        {CV_32F, {
            {CV_8U,  convertScaleData_<float    , uchar    >},
            {CV_16U, convertScaleData_<float    , ushort   >},
            {CV_32U, convertScaleData_<float    , uint     >},
            {CV_64U, convertScaleData_<float    , uint64_t >},
            {CV_8S,  convertScaleData_<float    , schar    >},
            {CV_16S, convertScaleData_<float    , short    >},
            {CV_32S, convertScaleData_<float    , int      >},
            {CV_64S, convertScaleData_<float    , int64_t  >},
            //{CV_16F, convertScaleData_<float    , float16_t>},
            {CV_32F, convertScaleData_<float    , float    >},
            {CV_64F, convertScaleData_<float    , double   >},
        }},
        {CV_64F, {
            {CV_8U,  convertScaleData_<double   , uchar    >},
            {CV_16U, convertScaleData_<double   , ushort   >},
            {CV_32U, convertScaleData_<double   , uint     >},
            {CV_64U, convertScaleData_<double   , uint64_t >},
            {CV_8S,  convertScaleData_<double   , schar    >},
            {CV_16S, convertScaleData_<double   , short    >},
            {CV_32S, convertScaleData_<double   , int      >},
            {CV_64S, convertScaleData_<double   , int64_t  >},
            //{CV_16F, convertScaleData_<double   , float16_t>},
            {CV_32F, convertScaleData_<double   , float    >},
            {CV_64F, convertScaleData_<double   , double   >},
        }},
    };

    ConvertScaleData func = map.at(CV_MAT_DEPTH(fromType)).at(CV_MAT_DEPTH(toType));
    CV_Assert( func != NULL );
    return func;
}

enum { HASH_SIZE0 = 8 };

static inline void copyElem(const uchar* from, uchar* to, size_t elemSize)
{
    size_t i;
    for( i = 0; i + sizeof(int) <= elemSize; i += sizeof(int) )
        *(int*)(to + i) = *(const int*)(from + i);
    for( ; i < elemSize; i++ )
        to[i] = from[i];
}

static inline bool isZeroElem(const uchar* data, size_t elemSize)
{
    size_t i;
    for( i = 0; i + sizeof(int) <= elemSize; i += sizeof(int) )
        if( *(int*)(data + i) != 0 )
            return false;
    for( ; i < elemSize; i++ )
        if( data[i] != 0 )
            return false;
    return true;
}

SparseMat::Hdr::Hdr( int _dims, const int* _sizes, int _type )
{
    refcount = 1;

    dims = _dims;
    valueOffset = (int)alignSize(sizeof(SparseMat::Node) - MAX_DIM*sizeof(int) +
                                 dims*sizeof(int), CV_ELEM_SIZE1(_type));
    nodeSize = alignSize(valueOffset +
        CV_ELEM_SIZE(_type), (int)sizeof(size_t));

    int i;
    for( i = 0; i < dims; i++ )
        size[i] = _sizes[i];
    for( ; i < CV_MAX_DIM; i++ )
        size[i] = 0;
    clear();
}

void SparseMat::Hdr::clear()
{
    hashtab.clear();
    hashtab.resize(HASH_SIZE0);
    pool.clear();
    pool.resize(nodeSize);
    nodeCount = freeList = 0;
}


SparseMat::SparseMat(const Mat& m)
: flags(MAGIC_VAL), hdr(0)
{
    create( m.dims, m.size, m.type() );

    int i, idx[CV_MAX_DIM] = {0}, d = m.dims, lastSize = m.size[d - 1];
    size_t esz = m.elemSize();
    const uchar* dptr = m.ptr();

    for(;;)
    {
        for( i = 0; i < lastSize; i++, dptr += esz )
        {
            if( isZeroElem(dptr, esz) )
                continue;
            idx[d-1] = i;
            uchar* to = newNode(idx, hash(idx));
            copyElem( dptr, to, esz );
        }

        for( i = d - 2; i >= 0; i-- )
        {
            dptr += m.step[i] - m.size[i+1]*m.step[i+1];
            if( ++idx[i] < m.size[i] )
                break;
            idx[i] = 0;
        }
        if( i < 0 )
            break;
    }
}

void SparseMat::create(int d, const int* _sizes, int _type)
{
    CV_Assert( _sizes && 0 < d && d <= CV_MAX_DIM );
    for( int i = 0; i < d; i++ )
        CV_Assert( _sizes[i] > 0 );
    _type = CV_MAT_TYPE(_type);
    if( hdr && _type == type() && hdr->dims == d && hdr->refcount == 1 )
    {
        int i;
        for( i = 0; i < d; i++ )
            if( _sizes[i] != hdr->size[i] )
                break;
        if( i == d )
        {
            clear();
            return;
        }
    }
    int _sizes_backup[CV_MAX_DIM]; // #5991
    if (hdr && _sizes == hdr->size)
    {
        for(int i = 0; i < d; i++ )
            _sizes_backup[i] = _sizes[i];
        _sizes = _sizes_backup;
    }
    release();
    flags = MAGIC_VAL | _type;
    hdr = new Hdr(d, _sizes, _type);
}

void SparseMat::copyTo( SparseMat& m ) const
{
    if( hdr == m.hdr )
        return;
    if( !hdr )
    {
        m.release();
        return;
    }
    m.create( hdr->dims, hdr->size, type() );
    SparseMatConstIterator from = begin();
    size_t N = nzcount(), esz = elemSize();

    for( size_t i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        uchar* to = m.newNode(n->idx, n->hashval);
        copyElem( from.ptr, to, esz );
    }
}

void SparseMat::copyTo( Mat& m ) const
{
    CV_Assert( hdr );
    int ndims = dims();
    m.create( ndims, hdr->size, type() );
    m = Scalar(0);

    SparseMatConstIterator from = begin();
    size_t N = nzcount(), esz = elemSize();

    for( size_t i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        copyElem( from.ptr, (ndims > 1 ? m.ptr(n->idx) : m.ptr(n->idx[0])), esz);
    }
}


void SparseMat::convertTo( SparseMat& m, int rtype, double alpha ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);
    if( hdr == m.hdr && rtype != type()  )
    {
        SparseMat temp;
        convertTo(temp, rtype, alpha);
        m = temp;
        return;
    }

    CV_Assert(hdr != 0);
    if( hdr != m.hdr )
        m.create( hdr->dims, hdr->size, rtype );

    SparseMatConstIterator from = begin();
    size_t N = nzcount();

    if( alpha == 1 )
    {
        ConvertData cvtfunc = getConvertElem(type(), rtype);
        for( size_t i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = hdr == m.hdr ? from.ptr : m.newNode(n->idx, n->hashval);
            cvtfunc( from.ptr, to, cn );
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleElem(type(), rtype);
        for( size_t i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = hdr == m.hdr ? from.ptr : m.newNode(n->idx, n->hashval);
            cvtfunc( from.ptr, to, cn, alpha, 0 );
        }
    }
}


void SparseMat::convertTo( Mat& m, int rtype, double alpha, double beta ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);

    CV_Assert( hdr );
    m.create( dims(), hdr->size, rtype );
    m = Scalar(beta);

    SparseMatConstIterator from = begin();
    size_t N = nzcount();

    if( alpha == 1 && beta == 0 )
    {
        ConvertData cvtfunc = getConvertElem(type(), rtype);
        for( size_t i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.ptr(n->idx);
            cvtfunc( from.ptr, to, cn );
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleElem(type(), rtype);
        for( size_t i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.ptr(n->idx);
            cvtfunc( from.ptr, to, cn, alpha, beta );
        }
    }
}

void SparseMat::clear()
{
    if( hdr )
        hdr->clear();
}

uchar* SparseMat::ptr(int i0, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 1 );
    size_t h = hashval ? *hashval : hash(i0);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0 };
        return newNode( idx, h );
    }
    return NULL;
}

uchar* SparseMat::ptr(int i0, int i1, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 2 );
    size_t h = hashval ? *hashval : hash(i0, i1);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 && elem->idx[1] == i1 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0, i1 };
        return newNode( idx, h );
    }
    return NULL;
}

uchar* SparseMat::ptr(int i0, int i1, int i2, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 3 );
    size_t h = hashval ? *hashval : hash(i0, i1, i2);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 &&
            elem->idx[1] == i1 && elem->idx[2] == i2 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0, i1, i2 };
        return newNode( idx, h );
    }
    return NULL;
}

uchar* SparseMat::ptr(const int* idx, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr );
    int i, d = hdr->dims;
    size_t h = hashval ? *hashval : hash(idx);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h )
        {
            for( i = 0; i < d; i++ )
                if( elem->idx[i] != idx[i] )
                    break;
            if( i == d )
                return &value<uchar>(elem);
        }
        nidx = elem->next;
    }

    return createMissing ? newNode(idx, h) : NULL;
}

void SparseMat::erase(int i0, int i1, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 2 );
    size_t h = hashval ? *hashval : hash(i0, i1);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 && elem->idx[1] == i1 )
            break;
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::erase(int i0, int i1, int i2, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 3 );
    size_t h = hashval ? *hashval : hash(i0, i1, i2);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 &&
            elem->idx[1] == i1 && elem->idx[2] == i2 )
            break;
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::erase(const int* idx, size_t* hashval)
{
    CV_Assert( hdr );
    int i, d = hdr->dims;
    size_t h = hashval ? *hashval : hash(idx);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h )
        {
            for( i = 0; i < d; i++ )
                if( elem->idx[i] != idx[i] )
                    break;
            if( i == d )
                break;
        }
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::resizeHashTab(size_t newsize)
{
    newsize = std::max(newsize, (size_t)8);
    if((newsize & (newsize-1)) != 0)
        newsize = (size_t)1 << cvCeil(std::log((double)newsize)/CV_LOG2);

    size_t hsize = hdr->hashtab.size();
    std::vector<size_t> _newh(newsize);
    size_t* newh = &_newh[0];
    for( size_t i = 0; i < newsize; i++ )
        newh[i] = 0;
    uchar* pool = &hdr->pool[0];
    for( size_t i = 0; i < hsize; i++ )
    {
        size_t nidx = hdr->hashtab[i];
        while( nidx )
        {
            Node* elem = (Node*)(pool + nidx);
            size_t next = elem->next;
            size_t newhidx = elem->hashval & (newsize - 1);
            elem->next = newh[newhidx];
            newh[newhidx] = nidx;
            nidx = next;
        }
    }
    hdr->hashtab = _newh;
}

uchar* SparseMat::newNode(const int* idx, size_t hashval)
{
    const int HASH_MAX_FILL_FACTOR=3;
    assert(hdr);
    size_t hsize = hdr->hashtab.size();
    if( ++hdr->nodeCount > hsize*HASH_MAX_FILL_FACTOR )
    {
        resizeHashTab(std::max(hsize*2, (size_t)8));
        hsize = hdr->hashtab.size();
    }

    if( !hdr->freeList )
    {
        size_t i, nsz = hdr->nodeSize, psize = hdr->pool.size(),
            newpsize = std::max(psize*3/2, 8*nsz);
        newpsize = (newpsize/nsz)*nsz;
        hdr->pool.resize(newpsize);
        uchar* pool = &hdr->pool[0];
        hdr->freeList = std::max(psize, nsz);
        for( i = hdr->freeList; i < newpsize - nsz; i += nsz )
            ((Node*)(pool + i))->next = i + nsz;
        ((Node*)(pool + i))->next = 0;
    }
    size_t nidx = hdr->freeList;
    Node* elem = (Node*)&hdr->pool[nidx];
    hdr->freeList = elem->next;
    elem->hashval = hashval;
    size_t hidx = hashval & (hsize - 1);
    elem->next = hdr->hashtab[hidx];
    hdr->hashtab[hidx] = nidx;

    int i, d = hdr->dims;
    for( i = 0; i < d; i++ )
        elem->idx[i] = idx[i];
    size_t esz = elemSize();
    uchar* p = &value<uchar>(elem);
    if( esz == sizeof(float) )
        *((float*)p) = 0.f;
    else if( esz == sizeof(double) )
        *((double*)p) = 0.;
    else
        memset(p, 0, esz);

    return p;
}


void SparseMat::removeNode(size_t hidx, size_t nidx, size_t previdx)
{
    Node* n = node(nidx);
    if( previdx )
    {
        Node* prev = node(previdx);
        prev->next = n->next;
    }
    else
        hdr->hashtab[hidx] = n->next;
    n->next = hdr->freeList;
    hdr->freeList = nidx;
    --hdr->nodeCount;
}

//
// Operations
//
double norm( const SparseMat& src, int normType )
{
    CV_INSTRUMENT_REGION();

    SparseMatConstIterator it = src.begin();

    size_t i, N = src.nzcount();
    normType &= NORM_TYPE_MASK;
    int type = src.type();
    double result = 0;

    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    if( type == CV_32F )
    {
        if( normType == NORM_INF )
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                result = std::max(result, std::abs((double)it.value<float>()));
            }
        else if( normType == NORM_L1 )
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                result += std::abs(it.value<float>());
            }
        else
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                double v = it.value<float>();
                result += v*v;
            }
    }
    else if( type == CV_64F )
    {
        if( normType == NORM_INF )
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                result = std::max(result, std::abs(it.value<double>()));
            }
        else if( normType == NORM_L1 )
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                result += std::abs(it.value<double>());
            }
        else
            for( i = 0; i < N; i++, ++it )
            {
                CV_Assert(it.ptr);
                double v = it.value<double>();
                result += v*v;
            }
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 32f and 64f are supported" );

    if( normType == NORM_L2 )
        result = std::sqrt(result);
    return result;
}

void minMaxLoc( const SparseMat& src, double* _minval, double* _maxval, int* _minidx, int* _maxidx )
{
    CV_INSTRUMENT_REGION();

    SparseMatConstIterator it = src.begin();
    size_t i, N = src.nzcount(), d = src.hdr ? src.hdr->dims : 0;
    int type = src.type();
    const int *minidx = 0, *maxidx = 0;

    if( type == CV_32F )
    {
        float minval = FLT_MAX, maxval = -FLT_MAX;
        for( i = 0; i < N; i++, ++it )
        {
            CV_Assert(it.ptr);
            float v = it.value<float>();
            if( v < minval )
            {
                minval = v;
                minidx = it.node()->idx;
            }
            if( v > maxval )
            {
                maxval = v;
                maxidx = it.node()->idx;
            }
        }
        if( _minval )
            *_minval = minval;
        if( _maxval )
            *_maxval = maxval;
    }
    else if( type == CV_64F )
    {
        double minval = DBL_MAX, maxval = -DBL_MAX;
        for( i = 0; i < N; i++, ++it )
        {
            CV_Assert(it.ptr);
            double v = it.value<double>();
            if( v < minval )
            {
                minval = v;
                minidx = it.node()->idx;
            }
            if( v > maxval )
            {
                maxval = v;
                maxidx = it.node()->idx;
            }
        }
        if( _minval )
            *_minval = minval;
        if( _maxval )
            *_maxval = maxval;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 32f and 64f are supported" );

    if( _minidx && minidx )
        for( i = 0; i < d; i++ )
            _minidx[i] = minidx[i];
    if( _maxidx && maxidx )
        for( i = 0; i < d; i++ )
            _maxidx[i] = maxidx[i];
}


void normalize( const SparseMat& src, SparseMat& dst, double a, int norm_type )
{
    CV_INSTRUMENT_REGION();

    double scale = 1;
    if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( src, norm_type );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );

    src.convertTo( dst, -1, scale );
}

} // cv::

//
// C-API glue
//
CvSparseMat* cvCreateSparseMat(const cv::SparseMat& sm)
{
    if( !sm.hdr || sm.hdr->dims > (int)cv::SparseMat::MAX_DIM)
        return 0;

    CvSparseMat* m = cvCreateSparseMat(sm.hdr->dims, sm.hdr->size, sm.type());

    cv::SparseMatConstIterator from = sm.begin();
    size_t i, N = sm.nzcount(), esz = sm.elemSize();

    for( i = 0; i < N; i++, ++from )
    {
        const cv::SparseMat::Node* n = from.node();
        uchar* to = cvPtrND(m, n->idx, 0, -2, 0);
        cv::copyElem(from.ptr, to, esz);
    }
    return m;
}

void CvSparseMat::copyToSparseMat(cv::SparseMat& m) const
{
    m.create( dims, &size[0], type );

    CvSparseMatIterator it;
    CvSparseNode* n = cvInitSparseMatIterator(this, &it);
    size_t esz = m.elemSize();

    for( ; n != 0; n = cvGetNextSparseNode(&it) )
    {
        const int* idx = CV_NODE_IDX(this, n);
        uchar* to = m.newNode(idx, m.hash(idx));
        cv::copyElem((const uchar*)CV_NODE_VAL(this, n), to, esz);
    }
}
