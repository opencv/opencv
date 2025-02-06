// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "persistence_impl.hpp"
#include "persistence_base64_encoding.hpp"

namespace cv
{

class base64::Base64ContextEmitter
{
public:
    explicit Base64ContextEmitter(cv::FileStorage::Impl& fs, bool needs_indent_)
            : file_storage(fs)
            , needs_indent(needs_indent_)
            , binary_buffer(BUFFER_LEN)
            , base64_buffer(base64_encode_buffer_size(BUFFER_LEN))
            , src_beg(0)
            , src_cur(0)
            , src_end(0)
    {
        src_beg = binary_buffer.data();
        src_end = src_beg + BUFFER_LEN;
        src_cur = src_beg;

        CV_Assert(fs.write_mode);

        if (needs_indent)
        {
            file_storage.flush();
        }
    }

    ~Base64ContextEmitter()
    {
        /* cleaning */
        if (src_cur != src_beg)
            flush();    /* encode the rest binary data to base64 buffer */
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

        if ( !needs_indent)
        {
            file_storage.puts((const char*)base64_buffer.data());
        }
        else
        {
            const char newline[] = "\n";
            char space[80];
            int ident = file_storage.write_stack.back().indent;
            memset(space, ' ', static_cast<int>(ident));
            space[ident] = '\0';

            file_storage.puts(space);
            file_storage.puts((const char*)base64_buffer.data());
            file_storage.puts(newline);
            file_storage.flush();
        }

        return true;
    }

private:
    /* because of Base64, we must keep its length a multiple of 3 */
    static const size_t BUFFER_LEN = 48U;
    // static_assert(BUFFER_LEN % 3 == 0, "BUFFER_LEN is invalid");

private:
    cv::FileStorage::Impl& file_storage;
    bool needs_indent;

    std::vector<uchar> binary_buffer;
    std::vector<uchar> base64_buffer;
    uchar * src_beg;
    uchar * src_cur;
    uchar * src_end;
};

std::string base64::make_base64_header(const char *dt) {
    std::ostringstream oss;
    oss << dt   << ' ';
    std::string buffer(oss.str());
    CV_Assert(buffer.size() < ::base64::HEADER_SIZE);

    buffer.reserve(::base64::HEADER_SIZE);
    while (buffer.size() < ::base64::HEADER_SIZE)
        buffer += ' ';

    return buffer;
}

size_t base64::base64_encode(const uint8_t *src, uint8_t *dst, size_t off, size_t cnt) {
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

int base64::icvCalcStructSize(const char *dt, int initial_size) {
    int size = cv::fs::calcElemSize( dt, initial_size );
    size_t elem_max_size = 0;
    for ( const char * type = dt; *type != '\0'; type++ ) {
        switch ( *type )
        {
            case 'u': { elem_max_size = std::max( elem_max_size, sizeof(uchar ) ); break; }
            case 'c': { elem_max_size = std::max( elem_max_size, sizeof(schar ) ); break; }
            case 'w': { elem_max_size = std::max( elem_max_size, sizeof(ushort) ); break; }
            case 's': { elem_max_size = std::max( elem_max_size, sizeof(short ) ); break; }
            case 'i': { elem_max_size = std::max( elem_max_size, sizeof(int   ) ); break; }
            case 'f': { elem_max_size = std::max( elem_max_size, sizeof(float ) ); break; }
            case 'd': { elem_max_size = std::max( elem_max_size, sizeof(double) ); break; }
            case 'I': { elem_max_size = std::max( elem_max_size, sizeof(int64_t)); break; }
            case 'U': { elem_max_size = std::max( elem_max_size, sizeof(uint64_t)); break; }
            default: break;
        }
    }
    size = cvAlign( size, static_cast<int>(elem_max_size) );
    return size;
}

size_t base64::base64_encode_buffer_size(size_t cnt, bool is_end_with_zero) {
    size_t additional = static_cast<size_t>(is_end_with_zero == true);
    return (cnt + 2U) / 3U * 4U + additional;
}

base64::Base64Writer::Base64Writer(cv::FileStorage::Impl& fs, bool can_indent)
        : emitter(new Base64ContextEmitter(fs, can_indent))
        , data_type_string()
{
    CV_Assert(fs.write_mode);
}

void base64::Base64Writer::write(const void* _data, size_t len, const char* dt)
{
    check_dt(dt);
    RawDataToBinaryConvertor convertor(_data, static_cast<int>(len), data_type_string);
    emitter->write(convertor);
}

template<typename _to_binary_convertor_t> inline
void base64::Base64Writer::write(_to_binary_convertor_t & convertor, const char* dt)
{
    check_dt(dt);
    emitter->write(convertor);
}

base64::Base64Writer::~Base64Writer()
{
    delete emitter;
}

void base64::Base64Writer::check_dt(const char* dt)
{
    if ( dt == 0 )
        CV_Error( cv::Error::StsBadArg, "Invalid \'dt\'." );
    else if (data_type_string.empty()) {
        data_type_string = dt;

        /* output header */
        std::string buffer = make_base64_header(dt);
        const uchar * beg = reinterpret_cast<const uchar *>(buffer.data());
        const uchar * end = beg + buffer.size();

        emitter->write(beg, end);
    } else if ( data_type_string != dt )
        CV_Error( cv::Error::StsBadArg, "\'dt\' does not match." );
}

base64::RawDataToBinaryConvertor::RawDataToBinaryConvertor(const void* src, int len, const std::string & dt)
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

    step = icvCalcStructSize(dt.c_str(), 0);
    end = beg + static_cast<size_t>(len);
}

inline  base64::RawDataToBinaryConvertor&  base64::RawDataToBinaryConvertor::operator >>(uchar * & dst)
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

inline  base64::RawDataToBinaryConvertor::operator bool() const
{
    return cur < end;
}

size_t base64::RawDataToBinaryConvertor::make_to_binary_funcs(const std::string &dt)
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
                case 'I':
                case 'U':
                    size = sizeof(uint64_t);
                    pack.func = to_binary<uint64_t>;
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

}
