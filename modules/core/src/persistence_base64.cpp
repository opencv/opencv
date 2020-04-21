// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "persistence.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/configuration.private.hpp>

namespace base64 {

typedef uchar uint8_t;

#if CHAR_BIT != 8
#error "`char` should be 8 bit."
#endif

uint8_t const base64_mapping[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

uint8_t const base64_padding = '=';

uint8_t const base64_demapping[] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0, 62,  0,  0,  0, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  0,  0,
    0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  0,  0,  0,  0,  0,  0, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51,  0,  0,  0,  0,
};

/*    `base64_demapping` above is generated in this way:
 *    `````````````````````````````````````````````````````````````````````
 *  std::string mapping((const char *)base64_mapping);
 *    for (auto ch = 0; ch < 127; ch++) {
 *        auto i = mapping.find(ch);
 *        printf("%3u, ", (i != std::string::npos ? i : 0));
 *    }
 *    putchar('\n');
 *    `````````````````````````````````````````````````````````````````````
 */

size_t base64_encode(uint8_t const * src, uint8_t * dst, size_t off, size_t cnt)
{
    if (!src || !dst || !cnt)
        return 0;

    /* initialize beginning and end */
    uint8_t       * dst_beg = dst;
    uint8_t       * dst_cur = dst_beg;

    uint8_t const * src_beg = src + off;
    uint8_t const * src_cur = src_beg;
    uint8_t const * src_end = src_cur + cnt / 3U * 3U;

    /* integer multiples part */
    while (src_cur < src_end) {
        uint8_t _2 = *src_cur++;
        uint8_t _1 = *src_cur++;
        uint8_t _0 = *src_cur++;
        *dst_cur++ = base64_mapping[ _2          >> 2U];
        *dst_cur++ = base64_mapping[(_1 & 0xF0U) >> 4U | (_2 & 0x03U) << 4U];
        *dst_cur++ = base64_mapping[(_0 & 0xC0U) >> 6U | (_1 & 0x0FU) << 2U];
        *dst_cur++ = base64_mapping[ _0 & 0x3FU];
    }

    /* remainder part */
    size_t rst = src_beg + cnt - src_cur;
    if (rst == 1U) {
        uint8_t _2 = *src_cur++;
        *dst_cur++ = base64_mapping[ _2          >> 2U];
        *dst_cur++ = base64_mapping[(_2 & 0x03U) << 4U];
    } else if (rst == 2U) {
        uint8_t _2 = *src_cur++;
        uint8_t _1 = *src_cur++;
        *dst_cur++ = base64_mapping[ _2          >> 2U];
        *dst_cur++ = base64_mapping[(_2 & 0x03U) << 4U | (_1 & 0xF0U) >> 4U];
        *dst_cur++ = base64_mapping[(_1 & 0x0FU) << 2U];
    }

    /* padding */
    switch (rst)
    {
    case 1U: *dst_cur++ = base64_padding;
        /* fallthrough */
    case 2U: *dst_cur++ = base64_padding;
        /* fallthrough */
    default: *dst_cur   = 0;
        break;
    }

    return static_cast<size_t>(dst_cur - dst_beg);
}

size_t base64_encode(char const * src, char * dst, size_t off, size_t cnt)
{
    if (cnt == 0U)
        cnt = std::strlen(src);

    return base64_encode
    (
        reinterpret_cast<uint8_t const *>(src),
        reinterpret_cast<uint8_t       *>(dst),
        off,
        cnt
    );
}

size_t base64_decode(uint8_t const * src, uint8_t * dst, size_t off, size_t cnt)
{
    /* check parameters */
    if (!src || !dst || !cnt)
        return 0U;
    if (cnt & 0x3U)
        return 0U;

    /* initialize beginning and end */
    uint8_t       * dst_beg = dst;
    uint8_t       * dst_cur = dst_beg;

    uint8_t const * src_beg = src + off;
    uint8_t const * src_cur = src_beg;
    uint8_t const * src_end = src_cur + cnt;

    /* start decoding */
    while (src_cur < src_end) {
        uint8_t d50 = base64_demapping[*src_cur++];
        uint8_t c50 = base64_demapping[*src_cur++];
        uint8_t b50 = base64_demapping[*src_cur++];
        uint8_t a50 = base64_demapping[*src_cur++];

        uint8_t b10 = b50 & 0x03U;
        uint8_t b52 = b50 & 0x3CU;
        uint8_t c30 = c50 & 0x0FU;
        uint8_t c54 = c50 & 0x30U;

        *dst_cur++ = (d50 << 2U) | (c54 >> 4U);
        *dst_cur++ = (c30 << 4U) | (b52 >> 2U);
        *dst_cur++ = (b10 << 6U) | (a50 >> 0U);
    }

    *dst_cur = 0;
    return size_t(dst_cur - dst_beg);
}

size_t base64_decode(char const * src, char * dst, size_t off, size_t cnt)
{
    if (cnt == 0U)
        cnt = std::strlen(src);

    return base64_decode
    (
        reinterpret_cast<uint8_t const *>(src),
        reinterpret_cast<uint8_t       *>(dst),
        off,
        cnt
    );
}

bool base64_valid(uint8_t const * src, size_t off, size_t cnt)
{
    /* check parameters */
    if (src == 0 || src + off == 0)
        return false;
    if (cnt == 0U)
        cnt = std::strlen(reinterpret_cast<char const *>(src));
    if (cnt == 0U)
        return false;
    if (cnt & 0x3U)
        return false;

    /* initialize beginning and end */
    uint8_t const * beg = src + off;
    uint8_t const * end = beg + cnt;

    /* skip padding */
    if (*(end - 1U) == base64_padding) {
        end--;
        if (*(end - 1U) == base64_padding)
            end--;
    }

    /* find illegal characters */
    for (uint8_t const * iter = beg; iter < end; iter++)
        if (*iter > 126U || (!base64_demapping[(uint8_t)*iter] && *iter != base64_mapping[0]))
            return false;

    return true;
}

bool base64_valid(char const * src, size_t off, size_t cnt)
{
    if (cnt == 0U)
        cnt = std::strlen(src);

    return base64_valid(reinterpret_cast<uint8_t const *>(src), off, cnt);
}

size_t base64_encode_buffer_size(size_t cnt, bool is_end_with_zero)
{
    size_t additional = static_cast<size_t>(is_end_with_zero == true);
    return (cnt + 2U) / 3U * 4U + additional;
}

size_t base64_decode_buffer_size(size_t cnt, bool is_end_with_zero)
{
    size_t additional = static_cast<size_t>(is_end_with_zero == true);
    return cnt / 4U * 3U + additional;
}

size_t base64_decode_buffer_size(size_t cnt, char  const * src, bool is_end_with_zero)
{
    return base64_decode_buffer_size(cnt, reinterpret_cast<uchar const *>(src), is_end_with_zero);
}

size_t base64_decode_buffer_size(size_t cnt, uchar const * src, bool is_end_with_zero)
{
    size_t padding_cnt = 0U;
    for (uchar const * ptr = src + cnt - 1U; *ptr == base64_padding; ptr--)
        padding_cnt ++;
    return base64_decode_buffer_size(cnt, is_end_with_zero) - padding_cnt;
}

/****************************************************************************
 * to_binary && binary_to
 ***************************************************************************/

template<typename _uint_t> inline size_t
to_binary(_uint_t val, uchar * cur)
{
    size_t delta = CHAR_BIT;
    size_t cnt = sizeof(_uint_t);
    while (cnt --> static_cast<size_t>(0U)) {
        *cur++ = static_cast<uchar>(val);
        val >>= delta;
    }
    return sizeof(_uint_t);
}

template<> inline size_t to_binary(double val, uchar * cur)
{
    Cv64suf bit64;
    bit64.f = val;
    return to_binary(bit64.u, cur);
}

template<> inline size_t to_binary(float val, uchar * cur)
{
    Cv32suf bit32;
    bit32.f = val;
    return to_binary(bit32.u, cur);
}

template<typename _primitive_t> inline size_t
to_binary(uchar const * val, uchar * cur)
{
    return to_binary<_primitive_t>(*reinterpret_cast<_primitive_t const *>(val), cur);
}


template<typename _uint_t> inline size_t
binary_to(uchar const * cur, _uint_t & val)
{
    val = static_cast<_uint_t>(0);
    for (size_t i = static_cast<size_t>(0U); i < sizeof(_uint_t); i++)
        val |= (static_cast<_uint_t>(*cur++) << (i * CHAR_BIT));
    return sizeof(_uint_t);
}

template<> inline size_t binary_to(uchar const * cur, double & val)
{
    Cv64suf bit64;
    binary_to(cur, bit64.u);
    val = bit64.f;
    return sizeof(val);
}

template<> inline size_t binary_to(uchar const * cur, float & val)
{
    Cv32suf bit32;
    binary_to(cur, bit32.u);
    val = bit32.f;
    return sizeof(val);
}

template<typename _primitive_t> inline size_t
binary_to(uchar const * cur, uchar * val)
{
    return binary_to<_primitive_t>(cur, *reinterpret_cast<_primitive_t *>(val));
}

/****************************************************************************
 * others
 ***************************************************************************/

std::string make_base64_header(const char * dt)
{
    std::ostringstream oss;
    oss << dt   << ' ';
    std::string buffer(oss.str());
    CV_Assert(buffer.size() < HEADER_SIZE);

    buffer.reserve(HEADER_SIZE);
    while (buffer.size() < HEADER_SIZE)
        buffer += ' ';

    return buffer;
}

bool read_base64_header(std::vector<char> const & header, std::string & dt)
{
    std::istringstream iss(header.data());
    return !!(iss >> dt);//the "std::basic_ios::operator bool" differs between C++98 and C++11. The "double not" syntax is portable and covers both cases with equivalent meaning
}

/****************************************************************************
 * Parser
 ***************************************************************************/

Base64ContextParser::Base64ContextParser(uchar * buffer, size_t size)
    : dst_cur(buffer)
    , dst_end(buffer + size)
    , base64_buffer(BUFFER_LEN)
    , src_beg(0)
    , src_cur(0)
    , src_end(0)
    , binary_buffer(base64_encode_buffer_size(BUFFER_LEN))
{
    src_beg = binary_buffer.data();
    src_cur = src_beg;
    src_end = src_beg + BUFFER_LEN;
}

Base64ContextParser::~Base64ContextParser()
{
    /* encode the rest binary data to base64 buffer */
    if (src_cur != src_beg)
        flush();
}

Base64ContextParser & Base64ContextParser::read(const uchar * beg, const uchar * end)
{
    if (beg >= end)
        return *this;

    while (beg < end) {
        /* collect binary data and copy to binary buffer */
        size_t len = std::min(end - beg, src_end - src_cur);
        std::memcpy(src_cur, beg, len);
        beg     += len;
        src_cur += len;

        if (src_cur >= src_end) {
            /* binary buffer is full. */
            /* decode it send result to dst */

            CV_Assert(flush());    /* check for base64_valid */
        }
    }

    return *this;
}

bool Base64ContextParser::flush()
{
    if ( !base64_valid(src_beg, 0U, src_cur - src_beg) )
        return false;

    if ( src_cur == src_beg )
        return true;

    uchar * buffer = binary_buffer.data();
    size_t len = base64_decode(src_beg, buffer, 0U, src_cur - src_beg);
    src_cur = src_beg;

    /* unexpected error */
    CV_Assert(len != 0);

    /* buffer is full */
    CV_Assert(dst_cur + len < dst_end);

    if (dst_cur + len < dst_end) {
        /* send data to dst */
        std::memcpy(dst_cur, buffer, len);
        dst_cur += len;
    }

    return true;
}

/****************************************************************************
 * Emitter
 ***************************************************************************/

/* A decorator for CvFileStorage
 * - no copyable
 * - not safe for now
 * - move constructor may be needed if C++11
 */
class Base64ContextEmitter
{
public:
    explicit Base64ContextEmitter(CvFileStorage * fs)
        : file_storage(fs)
        , binary_buffer(BUFFER_LEN)
        , base64_buffer(base64_encode_buffer_size(BUFFER_LEN))
        , src_beg(0)
        , src_cur(0)
        , src_end(0)
    {
        src_beg = binary_buffer.data();
        src_end = src_beg + BUFFER_LEN;
        src_cur = src_beg;

        CV_CHECK_OUTPUT_FILE_STORAGE(fs);

        if ( fs->fmt == CV_STORAGE_FORMAT_JSON )
        {
            /* clean and break buffer */
            *fs->buffer++ = '\0';
            ::icvPuts( fs, fs->buffer_start );
            fs->buffer = fs->buffer_start;
            memset( file_storage->buffer_start, 0, static_cast<int>(file_storage->space) );
            ::icvPuts( fs, "\"$base64$" );
        }
        else
        {
            ::icvFSFlush(file_storage);
        }
    }

    ~Base64ContextEmitter()
    {
        /* cleaning */
        if (src_cur != src_beg)
            flush();    /* encode the rest binary data to base64 buffer */

        if ( file_storage->fmt == CV_STORAGE_FORMAT_JSON )
        {
            /* clean and break buffer  */
            ::icvPuts(file_storage, "\"");
            file_storage->buffer = file_storage->buffer_start;
            ::icvFSFlush( file_storage );
            memset( file_storage->buffer_start, 0, static_cast<int>(file_storage->space) );
            file_storage->buffer = file_storage->buffer_start;
        }
    }

    Base64ContextEmitter & write(const uchar * beg, const uchar * end)
    {
        if (beg >= end)
            return *this;

        while (beg < end) {
            /* collect binary data and copy to binary buffer */
            size_t len = std::min(end - beg, src_end - src_cur);
           std::memcpy(src_cur, beg, len);
            beg     += len;
            src_cur += len;

            if (src_cur >= src_end) {
                /* binary buffer is full. */
                /* encode it to base64 and send result to fs */
                flush();
            }
        }

        return *this;
    }

    /*
     * a convertor must provide :
     * - `operator >> (uchar * & dst)` for writing current binary data to `dst` and moving to next data.
     * - `operator bool` for checking if current loaction is valid and not the end.
     */
    template<typename _to_binary_convertor_t> inline
    Base64ContextEmitter & write(_to_binary_convertor_t & convertor)
    {
        static const size_t BUFFER_MAX_LEN = 1024U;

        std::vector<uchar> buffer(BUFFER_MAX_LEN);
        uchar * beg = buffer.data();
        uchar * end = beg;

        while (convertor) {
            convertor >> end;
            write(beg, end);
            end = beg;
        }

        return *this;
    }

    bool flush()
    {
        /* control line width, so on. */
        size_t len = base64_encode(src_beg, base64_buffer.data(), 0U, src_cur - src_beg);
        if (len == 0U)
            return false;

        src_cur = src_beg;
        {
            if ( file_storage->fmt == CV_STORAGE_FORMAT_JSON )
            {
                ::icvPuts(file_storage, (const char*)base64_buffer.data());
            }
            else
            {
                const char newline[] = "\n";
                char space[80];
                int ident = file_storage->struct_indent;
                memset(space, ' ', static_cast<int>(ident));
                space[ident] = '\0';

                ::icvPuts(file_storage, space);
                ::icvPuts(file_storage, (const char*)base64_buffer.data());
                ::icvPuts(file_storage, newline);
                ::icvFSFlush(file_storage);
            }

        }

        return true;
    }

private:
    /* because of Base64, we must keep its length a multiple of 3 */
    static const size_t BUFFER_LEN = 48U;
    // static_assert(BUFFER_LEN % 3 == 0, "BUFFER_LEN is invalid");

private:
    CvFileStorage * file_storage;

    std::vector<uchar> binary_buffer;
    std::vector<uchar> base64_buffer;
    uchar * src_beg;
    uchar * src_cur;
    uchar * src_end;
};


class RawDataToBinaryConvertor
{
public:

    RawDataToBinaryConvertor(const void* src, int len, const std::string & dt)
        : beg(reinterpret_cast<const uchar *>(src))
        , cur(0)
        , end(0)
    {
        CV_Assert(src);
        CV_Assert(!dt.empty());
        CV_Assert(len > 0);

        /* calc step and to_binary_funcs */
        step_packed = make_to_binary_funcs(dt);

        end = beg;
        cur = beg;

        step = ::icvCalcStructSize(dt.c_str(), 0);
        end = beg + step * static_cast<size_t>(len);
    }

    inline RawDataToBinaryConvertor & operator >>(uchar * & dst)
    {
        CV_DbgAssert(*this);

        for (size_t i = 0U, n = to_binary_funcs.size(); i < n; i++) {
            elem_to_binary_t & pack = to_binary_funcs[i];
            pack.func(cur + pack.offset, dst + pack.offset_packed);
        }
        cur += step;
        dst += step_packed;

        return *this;
    }

    inline operator bool() const
    {
        return cur < end;
    }

private:
    typedef size_t(*to_binary_t)(const uchar *, uchar *);
    struct elem_to_binary_t
    {
        size_t      offset;
        size_t      offset_packed;
        to_binary_t func;
    };

private:
    size_t make_to_binary_funcs(const std::string &dt)
    {
        size_t cnt = 0;
        size_t offset = 0;
        size_t offset_packed = 0;
        char type = '\0';

        std::istringstream iss(dt);
        while (!iss.eof()) {
            if (!(iss >> cnt)) {
                iss.clear();
                cnt = 1;
            }
            CV_Assert(cnt > 0U);
            if (!(iss >> type))
                break;

            while (cnt-- > 0)
            {
                elem_to_binary_t pack;

                size_t size = 0;
                switch (type)
                {
                case 'u':
                case 'c':
                    size = sizeof(uchar);
                    pack.func = to_binary<uchar>;
                    break;
                case 'w':
                case 's':
                    size = sizeof(ushort);
                    pack.func = to_binary<ushort>;
                    break;
                case 'i':
                    size = sizeof(uint);
                    pack.func = to_binary<uint>;
                    break;
                case 'f':
                    size = sizeof(float);
                    pack.func = to_binary<float>;
                    break;
                case 'd':
                    size = sizeof(double);
                    pack.func = to_binary<double>;
                    break;
                case 'r':
                default:
                    CV_Error(cv::Error::StsError, "type is not supported");
                };

                offset = static_cast<size_t>(cvAlign(static_cast<int>(offset), static_cast<int>(size)));
                pack.offset = offset;
                offset += size;

                pack.offset_packed = offset_packed;
                offset_packed += size;

                to_binary_funcs.push_back(pack);
            }
        }

        CV_Assert(iss.eof());
        return offset_packed;
    }

private:
    const uchar * beg;
    const uchar * cur;
    const uchar * end;

    size_t step;
    size_t step_packed;
    std::vector<elem_to_binary_t> to_binary_funcs;
};

class BinaryToCvSeqConvertor
{
public:
    BinaryToCvSeqConvertor(CvFileStorage* fs, const uchar* src, size_t total_byte_size, const char* dt)
        : cur(src)
        , end(src + total_byte_size)
    {
        CV_Assert(src);
        CV_Assert(dt);
        CV_Assert(total_byte_size > 0);

        step = make_funcs(dt);  // calc binary_to_funcs
        functor_iter = binary_to_funcs.begin();

        if (total_byte_size % step != 0)
            CV_PARSE_ERROR("Total byte size not match elememt size");
    }

    inline BinaryToCvSeqConvertor & operator >> (CvFileNode & dst)
    {
        CV_DbgAssert(*this);

        /* get current data */
        union
        {
            uchar mem[sizeof(double)];
            uchar  u;
            char   b;
            ushort w;
            short  s;
            int    i;
            float  f;
            double d;
        } buffer; /* for GCC -Wstrict-aliasing */
        std::memset(buffer.mem, 0, sizeof(buffer));
        functor_iter->func(cur + functor_iter->offset_packed, buffer.mem);

        /* set node::data */
        switch (functor_iter->cv_type)
        {
        case CV_8U : { dst.data.i = cv::saturate_cast<int>   (buffer.u); break;}
        case CV_8S : { dst.data.i = cv::saturate_cast<int>   (buffer.b); break;}
        case CV_16U: { dst.data.i = cv::saturate_cast<int>   (buffer.w); break;}
        case CV_16S: { dst.data.i = cv::saturate_cast<int>   (buffer.s); break;}
        case CV_32S: { dst.data.i = cv::saturate_cast<int>   (buffer.i); break;}
        case CV_32F: { dst.data.f = cv::saturate_cast<double>(buffer.f); break;}
        case CV_64F: { dst.data.f = cv::saturate_cast<double>(buffer.d); break;}
        default: break;
        }

        /* set node::tag */
        switch (functor_iter->cv_type)
        {
        case CV_8U :
        case CV_8S :
        case CV_16U:
        case CV_16S:
        case CV_32S: { dst.tag = CV_NODE_INT; /*std::printf("%i,", dst.data.i);*/ break; }
        case CV_32F:
        case CV_64F: { dst.tag = CV_NODE_REAL; /*std::printf("%.1f,", dst.data.f);*/ break; }
        default: break;
        }

        /* check if end */
        if (++functor_iter == binary_to_funcs.end()) {
            functor_iter = binary_to_funcs.begin();
            cur += step;
        }

        return *this;
    }

    inline operator bool() const
    {
        return cur < end;
    }

private:
    typedef size_t(*binary_to_t)(uchar const *, uchar *);
    struct binary_to_filenode_t
    {
        size_t      cv_type;
        size_t      offset_packed;
        binary_to_t func;
    };

private:
    size_t make_funcs(const char* dt)
    {
        size_t cnt = 0;
        char type = '\0';
        size_t offset = 0;
        size_t offset_packed = 0;

        std::istringstream iss(dt);
        while (!iss.eof()) {
            if (!(iss >> cnt)) {
                iss.clear();
                cnt = 1;
            }
            CV_Assert(cnt > 0U);
            if (!(iss >> type))
                break;

            while (cnt-- > 0)
            {
                binary_to_filenode_t pack;

                /* set func and offset */
                size_t size = 0;
                switch (type)
                {
                case 'u':
                case 'c':
                    size      = sizeof(uchar);
                    pack.func = binary_to<uchar>;
                    break;
                case 'w':
                case 's':
                    size      = sizeof(ushort);
                    pack.func = binary_to<ushort>;
                    break;
                case 'i':
                    size      = sizeof(uint);
                    pack.func = binary_to<uint>;
                    break;
                case 'f':
                    size      = sizeof(float);
                    pack.func = binary_to<float>;
                    break;
                case 'd':
                    size      = sizeof(double);
                    pack.func = binary_to<double>;
                    break;
                case 'r':
                default:
                    CV_Error(cv::Error::StsError, "type is not supported");
                }; // need a better way for outputting error.

                offset = static_cast<size_t>(cvAlign(static_cast<int>(offset), static_cast<int>(size)));
                if (offset != offset_packed)
                {
                    static bool skip_message = cv::utils::getConfigurationParameterBool("OPENCV_PERSISTENCE_SKIP_PACKED_STRUCT_WARNING",
#ifdef _DEBUG
                            false
#else
                            true
#endif
                    );
                    if (!skip_message)
                    {
                        CV_LOG_WARNING(NULL, "Binary converter: struct storage layout has been changed in OpenCV 3.4.7. Alignment gaps has been removed from the storage containers. "
                                "Details: https://github.com/opencv/opencv/pull/15050"
                        );
                        skip_message = true;
                    }
                }
                offset += size;

                pack.offset_packed = offset_packed;
                offset_packed += size;

                /* set type */
                switch (type)
                {
                case 'u': { pack.cv_type = CV_8U ; break; }
                case 'c': { pack.cv_type = CV_8S ; break; }
                case 'w': { pack.cv_type = CV_16U; break; }
                case 's': { pack.cv_type = CV_16S; break; }
                case 'i': { pack.cv_type = CV_32S; break; }
                case 'f': { pack.cv_type = CV_32F; break; }
                case 'd': { pack.cv_type = CV_64F; break; }
                case 'r':
                default:
                    CV_Error(cv::Error::StsError, "type is not supported");
                } // need a better way for outputting error.

                binary_to_funcs.push_back(pack);
            }
        }

        CV_Assert(iss.eof());
        CV_Assert(binary_to_funcs.size());

        return offset_packed;
    }

private:

    const uchar * cur;
    const uchar * end;

    size_t step;
    std::vector<binary_to_filenode_t> binary_to_funcs;
    std::vector<binary_to_filenode_t>::iterator functor_iter;
};


/****************************************************************************
 * Wrapper
 ***************************************************************************/

Base64Writer::Base64Writer(::CvFileStorage * fs)
    : emitter(new Base64ContextEmitter(fs))
    , data_type_string()
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
}

void Base64Writer::write(const void* _data, size_t len, const char* dt)
{
    check_dt(dt);
    RawDataToBinaryConvertor convertor(_data, static_cast<int>(len), data_type_string);
    emitter->write(convertor);
}

template<typename _to_binary_convertor_t> inline
void Base64Writer::write(_to_binary_convertor_t & convertor, const char* dt)
{
    check_dt(dt);
    emitter->write(convertor);
}

Base64Writer::~Base64Writer()
{
    delete emitter;
}

void Base64Writer::check_dt(const char* dt)
{
    if ( dt == 0 )
        CV_Error( CV_StsBadArg, "Invalid \'dt\'." );
    else if (data_type_string.empty()) {
        data_type_string = dt;

        /* output header */
        std::string buffer = make_base64_header(dt);
        const uchar * beg = reinterpret_cast<const uchar *>(buffer.data());
        const uchar * end = beg + buffer.size();

        emitter->write(beg, end);
    } else if ( data_type_string != dt )
        CV_Error( CV_StsBadArg, "\'dt\' does not match." );
}


void make_seq(CvFileStorage* fs, const uchar* binary, size_t total_byte_size, const char * dt, ::CvSeq & seq)
{
    if (total_byte_size == 0)
        return;
    ::CvFileNode node;
    node.info = 0;
    BinaryToCvSeqConvertor convertor(fs, binary, total_byte_size, dt);
    while (convertor) {
        convertor >> node;
        cvSeqPush(&seq, &node);
    }
}

} // base64::

/****************************************************************************
 * Interface
 ***************************************************************************/

CV_IMPL void cvWriteRawDataBase64(::CvFileStorage* fs, const void* _data, int len, const char* dt)
{
    CV_Assert(fs);
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);

    check_if_write_struct_is_delayed( fs, true );

    if ( fs->state_of_writing_base64 == base64::fs::Uncertain )
    {
        switch_to_Base64_state( fs, base64::fs::InUse );
    }
    else if ( fs->state_of_writing_base64 != base64::fs::InUse )
    {
        CV_Error( CV_StsError, "Base64 should not be used at present." );
    }

    fs->base64_writer->write(_data, len, dt);
}
