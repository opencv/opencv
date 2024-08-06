// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include <list>
#include <tuple>
#include <unordered_map>
#include "zlib.h"
#include "stb_truetype.hpp"

namespace cv
{

#include "builtin_font_sans.h"
#include "builtin_font_italic.h"
#ifdef HAVE_UNIFONT
#include "builtin_font_uni.h"
#endif

typedef stbtt_fontinfo font_t;

/////////////////////// Some temporary stub for Harfbuzz API /////////////////////////

#ifndef HAVE_HARFBUZZ

typedef struct hb_glyph_position_t
{
    int x_advance;
    int y_advance;
    int x_offset;
    int y_offset;
} hb_glyph_position_t;

typedef struct hb_font_t
{
    int font_data;
} hb_font_t;

typedef struct hb_buffer_t
{
    int buf_data;
} hb_buffer_t;

typedef enum hb_direction_t
{
    HB_DIRECTION_INVALID = 0,
    HB_DIRECTION_LTR = 1,
    HB_DIRECTION_RTL = 2
} hb_direction_t;

typedef enum hb_script_t
{
    HB_SCRIPT_INVALID = -1,
    HB_SCRIPT_UNKNOWN = 0,
    HB_SCRIPT_COMMON = 1,
    HB_SCRIPT_INHERITED,
    HB_SCRIPT_LATIN,
    HB_SCRIPT_CYRILLIC,
    HB_SCRIPT_GREEK,

    HB_SCRIPT_HAN,
    HB_SCRIPT_HIRAGANA,

    HB_SCRIPT_ARABIC,
    HB_SCRIPT_HEBREW,
    HB_SCRIPT_SYRIAC,
    HB_SCRIPT_THAANA
} hb_script_t;

typedef struct hb_unicode_funcs_t
{
    int funcs_data;
} hb_unicode_funcs_t;

static hb_buffer_t* hb_buffer_create() { return 0; }
static void hb_buffer_destroy(hb_buffer_t*) {}
static void hb_buffer_guess_segment_properties(hb_buffer_t *) {}
static hb_unicode_funcs_t* hb_unicode_funcs_get_default() { return 0; }

#endif

//////////////////////////////////////////////////////////////////////////////////////

typedef struct BuiltinFontData
{
    const uchar* gzdata;
    size_t size;
    const char* name;
    double sf;
    bool italic;
} BuiltinFontData;

enum
{
    BUILTIN_FONTS_NUM = 2
#ifdef HAVE_UNIFONT
        +1
#endif
};

static BuiltinFontData builtinFontData[BUILTIN_FONTS_NUM+1] =
{
    {OcvBuiltinFontSans, sizeof(OcvBuiltinFontSans), "sans", 1.0, false},
    {OcvBuiltinFontItalic, sizeof(OcvBuiltinFontItalic), "italic", 1.0, true},
#ifdef HAVE_UNIFONT
    {OcvBuiltinFontUni, sizeof(OcvBuiltinFontUni), "uni", 1.0, true},
#endif
    {0, 0, 0, 0.0, false}
};

static bool inflate(const void* src, size_t srclen, std::vector<uchar>& dst)
{
    std::vector<uchar> newdst((size_t)(srclen*2.5));
    std::swap(dst, newdst); // make sure we deallocated all the unused space that would be wasted otherwise
    for(int attempts = 0; attempts < 5; attempts++)
    {
        z_stream strm = {};
        strm.total_in = strm.avail_in  = (uInt)srclen;
        strm.total_out = strm.avail_out = (uInt)dst.size();
        strm.next_in = (Bytef*)src;
        strm.next_out = (Bytef*)&dst[0];

        int err = inflateInit2(&strm, (15 + 32)); //15 window bits, and the +32 tells zlib to to detect if using gzip or zlib
        if (err == Z_OK)
        {
            err = inflate(&strm, Z_FINISH);
            inflateEnd(&strm);
            if (err == Z_STREAM_END)
            {
                dst.resize((size_t)strm.total_out);
                return true;
            }
            else
                dst.resize(dst.size()*3/2);
        }
        else
        {
            inflateEnd(&strm);
            return false;
        }
    }
    return false;
}

struct FontFace::Impl {
    Impl()
    {
        initParams();
        ttface = 0;
        hb_font = 0;
        scalefactor = 1.0;
        italic = false;
    }

    ~Impl()
    {
        deleteFont();
    }

    void deleteFont()
    {
        stbtt_ReleaseFont(&ttface);
        currname.clear();
        initParams();
    }

    void initParams()
    {
        currweight = -1;
        currsize = -1;
        scale = 1;
    }

    bool setStd(const BuiltinFontData& fontdata)
    {
        if(fontdata.size <= 1)
            return false;
        if(ttface == 0 || currname != fontdata.name)
        {
            deleteFont();
            if(!inflate(fontdata.gzdata, fontdata.size, fontbuf))
                return false;
            ttface = stbtt_CreateFont(&fontbuf[0], (unsigned)fontbuf.size(), stbtt_GetFontOffsetForIndex(&fontbuf[0],0));
            if (!ttface)
                return false;
        }
        currname = fontdata.name;
        scalefactor = fontdata.sf;
        italic = fontdata.italic;
        initParams();
        return true;
    }

    bool set(const String& fontname)
    {
        CV_Assert(!fontname.empty());

        if(ttface != 0 && fontname == currname)
            return true;

        deleteFont();

        const char* fntname = fontname.c_str();
        size_t fntnamelen = fontname.size();
        FILE* f = fopen(fntname, "rb");
        if (!f)
            return false;
        fseek(f, 0, SEEK_END);
        long sz0 = ftell(f), sz1 = 0;
        std::vector<uchar> srcdata(sz0);
        if (sz0 > 0)
        {
            fseek(f, 0, SEEK_SET);
            sz1 = (long)fread(&srcdata[0], 1, sz0, f);
        }
        fclose(f);
        if (sz0 == 0 || sz1 != sz0)
            return false;
        if (fntnamelen > 3 && strcmp(fntname + fntnamelen - 3, ".gz") == 0)
        {
            if (!inflate(&srcdata[0], srcdata.size(), fontbuf))
                return false;
        }
        else
        {
            fontbuf.resize(srcdata.size());
            std::copy(srcdata.begin(), srcdata.end(), fontbuf.begin());
        }
        ttface = stbtt_CreateFont(&fontbuf[0], (unsigned)fontbuf.size(), stbtt_GetFontOffsetForIndex(&fontbuf[0],0));
        if (!ttface)
            return false;
        currname = fontname;
        initParams();
        return true;
    }

    bool setParams(int size, int weight)
    {
        if (ttface == 0)
            return false;
        if (std::abs(size - currsize) < 1e-3 &&
            (weight == currweight || weight == 0))
            return true;

        if (weight != currweight && weight != 0) {
            int params[] = {STBTT_FOURCC('w','g','h','t'), weight};
            if (!stbtt_SetInstance(ttface, params, 1, 0))
                return false;
            currweight = weight;
        }

        if(size != currsize)
            scale = stbtt_ScaleForPixelHeightNoDesc(ttface, (float)size);

        currsize = size;
        return true;
    }

    String currname;
    double scalefactor;
    bool italic;
    int currsize;
    int currweight;
    float scale;
    stbtt_fontinfo* ttface;
    hb_font_t* hb_font;
    std::vector<uchar> fontbuf;
};

struct GlyphCacheKey
{
    GlyphCacheKey()
    {
        ttface = 0;
        size = 0;
        weight = 0;
    }
    GlyphCacheKey(font_t* ttface_, int index_, double size_, int weight_, float scale_)
    {
        ttface = ttface_;
        glyph_index = index_;
        size = cvRound(size_*256);
        weight = weight_;
        scale = scale_;
    }

    font_t* ttface;
    int glyph_index;
    int size;
    int weight;
    float scale;
};

static bool operator == (const GlyphCacheKey& k1, const GlyphCacheKey& k2)
{
    return k1.ttface == k2.ttface && k1.glyph_index == k2.glyph_index &&
           k1.size == k2.size && k1.weight == k2.weight && k1.scale == k2.scale;
}

static size_t hash_seq(const size_t* hashvals, size_t n)
{
    size_t h = (size_t)-1;
    const int hsz = (int)(sizeof(h)*8);
    for( size_t i = 0; i < n; i++ )
    {
        h = (h >> 5) ^ (h << (hsz - 5)) ^ hashvals[i];
    }
    return h;
}

struct GlyphCacheHash
{
    size_t operator()(const GlyphCacheKey& key) const noexcept
    {
        size_t hs[] = {(size_t)(void*)key.ttface, (size_t)key.glyph_index,
            (size_t)key.size, (size_t)key.weight}; // do not include scale, because it's completely defined by size
        return hash_seq(hs, sizeof(hs)/sizeof(hs[0]));
    }
};

struct GlyphCacheVal
{
    GlyphCacheVal()
    {
        crc = 0;
        width = height = linegap = ascent = horiBearingX = horiBearingY = 0;
    }

    std::vector<uchar> rlebuf;
    Rect bbox;
    unsigned crc;
    int width;
    int height;
    Point advance;
    int linegap, ascent;
    int horiBearingX;
    int horiBearingY;
};

typedef std::pair<GlyphCacheKey, GlyphCacheVal> GlyphCacheEntry;
typedef std::list<GlyphCacheEntry>::iterator GlyphCacheEntryIt;

struct TextSegment
{
    TextSegment()
    {
        ttface = 0;
        fontidx = 0;
        start = end = 0;
        script = HB_SCRIPT_UNKNOWN;
        dir = HB_DIRECTION_LTR;
    }
    TextSegment(font_t* ttface_, int fontidx_, int start_, int end_, hb_script_t script_, hb_direction_t dir_)
    {
        ttface = ttface_;
        fontidx = fontidx_;
        start = start_;
        end = end_;
        script = script_;
        dir = dir_;
    }
    font_t* ttface;
    int fontidx;
    int start, end;
    hb_script_t script;
    hb_direction_t dir;
};

struct FontGlyph
{
    FontGlyph() { ttface = 0; index = -1; scale = 1.f; }
    FontGlyph(font_t* ttface_, int glyph_index_, const hb_glyph_position_t& pos_, float scale_)
    {
        ttface = ttface_;
        index = glyph_index_;
        pos = pos_;
        scale = scale_;
    }
    font_t* ttface;
    int index;
    float scale;
    hb_glyph_position_t pos;
};


class FontRenderEngine
{
public:
    enum { MAX_CACHED_GLYPH_SIZE = 128, MAX_CACHE_SIZE=2048 };
    FontRenderEngine()
    {
        hb_buf = hb_buffer_create();
        hb_buffer_guess_segment_properties(hb_buf);
        hb_uni_funcs = hb_unicode_funcs_get_default();
        max_cache_size = (size_t)MAX_CACHE_SIZE;
        glyph_buf = 0;
        glyph_bufsz = 0;
    }

    ~FontRenderEngine()
    {
        hb_buffer_destroy(hb_buf);
        for(int i = 0; i < BUILTIN_FONTS_NUM; i++)
            builtin_ffaces[i] = FontFace();
        if (glyph_buf)
            free(glyph_buf);
        glyph_buf = 0;
        glyph_bufsz = 0;
    }

    void addToCache(const GlyphCacheKey& key, const GlyphCacheVal& val)
    {
        if(glyph_cache.size() == max_cache_size)
        {
            GlyphCacheEntryIt last = all_cached.end();
            --last;
            glyph_cache.erase(last->first);
            all_cached.pop_back();
        }

        all_cached.push_front(std::make_pair(key, val));
        GlyphCacheEntryIt first = all_cached.begin();
        glyph_cache.insert(std::make_pair(first->first, first));
    }

    GlyphCacheVal* findCachedGlyph(const GlyphCacheKey& key)
    {
        auto it = glyph_cache.find(key);
        if(it == glyph_cache.end())
            return 0;
        all_cached.splice(all_cached.begin(), all_cached, it->second);
        return &it->second->second;
    }

    FontFace& getStdFontFace(int i)
    {
        CV_Assert(i >= 0 && i < BUILTIN_FONTS_NUM);
        builtin_ffaces[i]->setStd(builtinFontData[i]);
        return builtin_ffaces[i];
    }

    Point putText_( Mat& img, Size imgsize, const String& str_, Point org,
                    const uchar* color, FontFace& fontface, int size,
                    int weight, PutTextFlags flags, Range wrapRange, bool render,
                    Rect* bbox );

protected:
    FontFace builtin_ffaces[BUILTIN_FONTS_NUM];
    bool builtin_ffaces_initialized;

    hb_unicode_funcs_t* hb_uni_funcs;

    // LRU cache of glyphs
    GlyphCacheVal new_cached;
    std::list<GlyphCacheEntry> all_cached;
    std::unordered_map<GlyphCacheKey, GlyphCacheEntryIt, GlyphCacheHash> glyph_cache;
    size_t max_cache_size;

    hb_buffer_t* hb_buf;
    std::vector<unsigned> u32buf;
    std::vector<TextSegment> segments;
    std::vector<FontGlyph> glyphs;
    std::vector<uchar> pixbuf;
    uchar* glyph_buf;
    int glyph_bufsz;
};

thread_local FontRenderEngine fontRenderEngine;

FontFace::FontFace() { impl = makePtr<Impl>(); }
FontFace::FontFace(const String& fontname)
{
    impl = makePtr<Impl>();
    set(fontname);
}

bool FontFace::set(const String& fontname_)
{
    String fontname = fontname_;
    if(fontname.empty())
        fontname = "sans";
    if(impl->ttface != 0 && impl->currname == fontname)
        return true;
    int i = 0;
    for( ; i < BUILTIN_FONTS_NUM; i++ )
    {
        if( builtinFontData[i].name == fontname && builtinFontData[i].size > 1 )
            break;
    }
    if( i >= BUILTIN_FONTS_NUM )
        i = -1;

    FontRenderEngine& engine = fontRenderEngine;

    bool ok = false;
    if( i >= 0 )
    {
        FontFace& builtin_fface = engine.getStdFontFace(i);
        if(builtin_fface.impl->ttface)
        {
            impl = builtin_fface.impl;
            ok = true;
        }
    }
    else
    {
        if(impl->ttface != 0)
            impl = makePtr<Impl>();
        ok = impl->set(fontname);
    }
    return ok;
}

bool FontFace::getBuiltinFontData(const String& fontname_,
                                  const uchar*& data, size_t& size)
{
    String fontname = fontname_;
    if(fontname.empty())
        fontname = "sans";
    data = 0;
    size = 0;
    for(int i = 0; i < BUILTIN_FONTS_NUM; i++)
    {
        if(builtinFontData[i].name == fontname && builtinFontData[i].size > 1)
        {
            FontFace& builtin_fface = fontRenderEngine.getStdFontFace(i);
            if( builtin_fface.impl->ttface )
            {
                std::vector<uchar>& fbuf = builtin_fface.impl->fontbuf;
                size = fbuf.size();
                data = size > 0 ? &fbuf[0] : 0;
                return size > 0;
            }
            return false;
        }
    }
    return false;
}

String FontFace::getName() const { return impl->currname; }
FontFace::Impl* FontFace::operator -> () { return impl.get(); }
FontFace::~FontFace() {}

bool FontFace::setInstance(const std::vector<int>& params)
{
    if (params.empty())
        return true;
    if (!impl->ttface)
        return false;
    CV_Assert(params.size() % 2 == 0);
    return stbtt_SetInstance(impl->ttface, &params[0], (int)(params.size()/2), 1) > 0;
}

bool FontFace::getInstance(std::vector<int>& params) const
{
    if (!impl->ttface)
        return false;

    stbtt_axisinfo axes[STBTT_MAX_AXES];
    int i, naxes = stbtt_GetInstance(impl->ttface, axes, STBTT_MAX_AXES);
    params.resize(naxes*2);

    for( i = 0; i < naxes; i++ )
    {
        int tag = axes[i].tag;
        params[i*2] = CV_FOURCC((char)(tag >> 24), (char)(tag >> 16), (char)(tag >> 8), (char)tag);
        params[i*2+1] = axes[i].currval;
    }
    return naxes > 0;
}

static unsigned calccrc(const std::vector<uchar>& buf)
{
    unsigned crc = (unsigned)-1;
    size_t i, n = buf.size();
    for(i = 0; i < n; i++)
        crc = (crc << 5) ^ (crc >> 27) ^ buf[i];
    return crc;
}

static void
compressCharacter(const uchar* bitmap_buf,
                  int bitmap_step, Size bitmap_size,
                  std::vector<uchar>& rlebuf,
                  Rect& roi, unsigned* crc)
{
    const int RLE_MAX = 255;
    int width = bitmap_size.width, height = bitmap_size.height;
    int left = width-1, right = 0, top = height-1, bottom = 0;
    bool global_have_nz = false;
    for( int i = 0; i < height; i++ )
    {
        bool have_nz = false;
        for( int j = 0; j < width; j++ )
        {
            if(bitmap_buf[i*bitmap_step + j] != 0)
            {
                left = min(left, j);
                right = max(right, j);
                have_nz = true;
            }
        }
        if( have_nz )
        {
            global_have_nz |= have_nz;
            top = min(top, i);
            bottom = max(bottom, i);
        }
    }
    // all 0's
    if(!global_have_nz)
    {
        roi = Rect(0, 0, 0, 0);
        rlebuf.clear();
        if(crc)
            *crc = calccrc(rlebuf);
        return;
    }
    roi = Rect(left, top, right - left + 1, bottom - top + 1);
    bool rle_mode = true;
    int k = 0, count = 0;
    int count_pos = -1;
    int bufsz = roi.width*roi.height*2+256;
    rlebuf.resize(bufsz);
    uchar* buf = &rlebuf[0];
    int prev = 0, x = 0;

    //  c b
    //  a x
    for( int i = top; i <= bottom; i++ )
    {
        for( int j = left; j <= right; j++ )
        {
            const uchar* ptr = bitmap_buf + i*bitmap_step + j;
            prev = x;
            x = *ptr;
            int a = 0, b = 0, c = 0;
            if( i > top )
            {
                b = ptr[-bitmap_step];
                if( j > left )
                {
                    a = ptr[-1];
                    c = ptr[-1-bitmap_step];
                }
            }
            else if( j > left )
                a = ptr[-1];
            int predicted = a + b - c;
            if( x == predicted )
            {
                if(rle_mode) ++count;
                else
                {
                    if(count > 0)
                        buf[count_pos] = (uchar)count;
                    count = 1;
                    rle_mode = true;
                }
            }
            else if( !rle_mode )
            {
                if(count == RLE_MAX)
                {
                    buf[count_pos] = (uchar)count;
                    buf[k++] = 0;
                    count_pos = k++;
                    count = 0;
                }
                buf[k++] = (uchar)x;
                count++;
            }
            else if( count == 1 && count_pos >= 0 && buf[count_pos] < RLE_MAX-1 )
            {
                count = buf[count_pos] + 2;
                buf[k++] = (uchar)prev;
                buf[k++] = (uchar)x;
                rle_mode = false;
            }
            else
            {
                if(count == 0)
                    buf[k++] = 0;
                while(count > 0)
                {
                    int dcount = min(count, RLE_MAX);
                    count -= dcount;
                    buf[k++] = (uchar)dcount;
                    if(count > 0)
                        buf[k++] = 0;
                }
                count_pos = k++;
                buf[k++] = (uchar)x;
                count = 1;
                rle_mode = false;
            }
        }
    }

    if(rle_mode)
    {
        while(count > 0)
        {
            int dcount = min(count, RLE_MAX);
            count -= dcount;
            buf[k++] = (uchar)dcount;
            if(count > 0)
                buf[k++] = 0;
        }
    }
    else buf[count_pos] = (uchar)count;
    rlebuf.resize(k);
    if(crc)
        *crc = calccrc(rlebuf);
}

static bool
decompressCharacter(const std::vector<uchar>& rlebuf, Rect r,
                    std::vector<uchar>& pixbuf, const unsigned* crc)
{
    if(crc)
        CV_Assert(*crc == calccrc(rlebuf));

    int width = r.width, height = r.height;
    int i = 0, j = 0, k = 0;
    int bufsz = (int)rlebuf.size();
    const uchar* buf = bufsz > 0 ? &rlebuf[0] : 0;

    pixbuf.resize(width*height);
    uchar* pixels = width*height > 0 ? &pixbuf[0] : 0;

    if(bufsz == 0)
    {
        if(width > 0 && height > 0)
            memset(pixels, 0, width*height);
        return true;
    }

    while( k < bufsz )
    {
        int rle_count = buf[k++];
        for( ; rle_count > 0; rle_count-- )
        {
            int a = 0, b = 0, c = 0;
            uchar* ptr = pixels + i*width + j;
            if( i > 0 )
            {
                b = ptr[-width];
                if( j > 0 )
                {
                    a = ptr[-1];
                    c = ptr[-width-1];
                }
            }
            else if( j > 0 )
                a = ptr[-1];
            int pred = a + b - c;
            ptr[0] = (uchar)pred;
            if( ++j >= width )
            {
                j = 0;
                if( ++i >= height )
                    return true;
            }
        }
        if( k >= bufsz )
            return false;
        int nz_end = buf[k++];
        nz_end += k;
        if( nz_end > bufsz )
            return false;
        for( ; k < nz_end; k++ )
        {
            pixels[i*width + j] = buf[k];
            if( ++j >= width )
            {
                j = 0;
                if( ++i >= height )
                    return true;
            }
        }
    }
    return false;
}

static void drawCharacter(
    Mat& img, const uchar* color,
    const uchar* bitmap_buf,
    int bitmap_step, Size bitmap_size,
    int x0, int y0, bool bottom_left )
{
    int nch = img.channels();
    int bw = bitmap_size.width, bh = bitmap_size.height;
    int rows = img.rows, cols = img.cols;
    uchar b = color[0], g = color[1], r = color[2];

    if(x0 >= cols || x0 + bw <= 0)
        return;

    // for simplicity, we assume that `bitmap->pixel_mode'
    // is `FT_PIXEL_MODE_GRAY' (i.e., not a bitmap font)
    for( int dy = 0; dy < bh; dy++ )
    {
        int y = y0 + dy*(bottom_left ? -1 : 1);
        if( y < 0 || y >= rows )
            continue;
        uchar* imgptr0 = img.ptr<uchar>(y);
        for( int dx = 0; dx < bw; dx++ )
        {
            int x = x0 + dx;
            if( x < 0 || x >= cols )
                continue;
            uchar* imgptr = imgptr0 + x*nch;
            uchar alpha = bitmap_buf[dy*bitmap_step + dx];
            if(alpha == 0)
                continue;
            if( nch == 3 )
            {
                uchar b1 = (uchar)((imgptr[0]*(255 - alpha) + b*alpha + 127)/255);
                uchar g1 = (uchar)((imgptr[1]*(255 - alpha) + g*alpha + 127)/255);
                uchar r1 = (uchar)((imgptr[2]*(255 - alpha) + r*alpha + 127)/255);
                imgptr[0] = b1;
                imgptr[1] = g1;
                imgptr[2] = r1;
            }
            else if(nch == 1)
            {
                uchar b1 = (uchar)((imgptr[0]*(255 - alpha) + b*alpha + 127)/255);
                imgptr[0] = b1;
            }
            else
            {
                uchar b1 = (uchar)((imgptr[0]*(255 - alpha) + b*alpha + 127)/255);
                uchar g1 = (uchar)((imgptr[1]*(255 - alpha) + g*alpha + 127)/255);
                uchar r1 = (uchar)((imgptr[2]*(255 - alpha) + r*alpha + 127)/255);
                imgptr[0] = b1;
                imgptr[1] = g1;
                imgptr[2] = r1;
                imgptr[3] = alpha;
            }
        }
    }
}

#ifdef HAVE_HARFBUZZ
//by amarullz from https://stackoverflow.com/questions/5423960/how-can-i-recognize-rtl-strings-in-c
static bool isRightToLeft(unsigned c)
{
    if(c < 0x5BE)
        return false;

    return
    ((c==0x05BE)||(c==0x05C0)||(c==0x05C3)||(c==0x05C6)||
    ((c>=0x05D0)&&(c<=0x05F4))||
    (c==0x0608)||(c==0x060B)||(c==0x060D)||
    ((c>=0x061B)&&(c<=0x064A))||
    ((c>=0x066D)&&(c<=0x066F))||
    ((c>=0x0671)&&(c<=0x06D5))||
    ((c>=0x06E5)&&(c<=0x06E6))||
    ((c>=0x06EE)&&(c<=0x06EF))||
    ((c>=0x06FA)&&(c<=0x0710))||
    ((c>=0x0712)&&(c<=0x072F))||
    ((c>=0x074D)&&(c<=0x07A5))||
    ((c>=0x07B1)&&(c<=0x07EA))||
    ((c>=0x07F4)&&(c<=0x07F5))||
    ((c>=0x07FA)&&(c<=0x0815))||
    (c==0x081A)||(c==0x0824)||(c==0x0828)||
    ((c>=0x0830)&&(c<=0x0858))||
    ((c>=0x085E)&&(c<=0x08AC))||
    (c==0x200F)||(c==0xFB1D)||
    ((c>=0xFB1F)&&(c<=0xFB28))||
    ((c>=0xFB2A)&&(c<=0xFD3D))||
    ((c>=0xFD50)&&(c<=0xFDFC))||
    ((c>=0xFE70)&&(c<=0xFEFC))||
    ((c>=0x10800)&&(c<=0x1091B))||
    ((c>=0x10920)&&(c<=0x10A00))||
    ((c>=0x10A10)&&(c<=0x10A33))||
    ((c>=0x10A40)&&(c<=0x10B35))||
    ((c>=0x10B40)&&(c<=0x10C48))||
    ((c>=0x1EE00)&&(c<=0x1EEBB)));
}
#endif

Point FontRenderEngine::putText_(
    Mat& img, Size imgsize, const String& str_, Point org,
    const uchar* color, FontFace& fontface, int size,
    int weight_, PutTextFlags flags, Range wrapRange,
    bool render, Rect* bbox_ )
{
    bool bottom_left = (flags & PUT_TEXT_ORIGIN_BL) != 0;
    int saved_weights[BUILTIN_FONTS_NUM+1]={0};
    int weight = weight_*65536;

    if(fontface.getName().empty())
        fontface.set("sans");
    if(!fontface->ttface)
        CV_Error(Error::StsError, "No available fonts for putText()");

    Point pen = org;
    int i, j, len = (int)str_.size();
    bool wrap = (flags & PUT_TEXT_WRAP) != 0;
    int alignment = flags & PUT_TEXT_ALIGN_MASK;
    int x0 = std::min(wrapRange.start, wrapRange.end);
    int x1 = std::max(wrapRange.start, wrapRange.end);

    if(x0 == 0 && x1 == 0)
    {
        if(alignment == PUT_TEXT_ALIGN_RIGHT)
        {
            x0 = 0;
            x1 = org.x;
        }
        else
        {
            x0 = org.x;
            x1 = imgsize.width > 0 ? imgsize.width : INT_MAX;
        }
    }

    if(x0 >= x1 || len == 0)
    {
        if(bbox_)
            *bbox_ = Rect(org.x, org.y, 0, 0);
        return org;
    }

    saved_weights[BUILTIN_FONTS_NUM] = stbtt_GetWeight(fontface->ttface);
    fontface->setParams(size, weight);

    for(j = 0; j < BUILTIN_FONTS_NUM; j++)
    {
        FontFace& fface = builtin_ffaces[j];
        if(!builtin_ffaces_initialized)
            fface.set(builtinFontData[j].name);

        if (fface->ttface) {
            saved_weights[j] = stbtt_GetWeight(fface->ttface);
            fface->setParams(size, weight);
        }
    }
    builtin_ffaces_initialized = true;

    if(alignment == PUT_TEXT_ALIGN_RIGHT)
        std::swap(x0, x1);

    int alignSign = alignment == PUT_TEXT_ALIGN_RIGHT ? -1 : 1;

    // text size computing algorithm is adopted from G-API module, ft_render.cpp.
    const char* str = &str_[0];
    int max_dy = 0, max_baseline = 0;
    int max_x = INT_MIN, min_x = INT_MAX;
    bool wrapped = false;

    // step 1. convert UTF8 to UTF32
    u32buf.clear();
    for( i = 0; i < len; i++ )
    {
        uchar ch = (uchar)str[i];
        unsigned charcode;
        if( ch <= 127 )
            charcode = ch;
        else if( ch <= 223 && i+1 < len && (str[i+1] & 0xc0) == 0x80) {
            charcode = ((ch & 31) << 6) | (str[i+1] & 63);
            i++;
        }
        else if( ch <= 239 && i+2 < len && (str[i+1] & 0xc0) == 0x80 && (str[i+2] & 0xc0) == 0x80) {
            charcode = ((ch & 15) << 12) | ((str[i+1] & 63) << 6) | (str[i+2] & 63);
            i += 2;
        }
        else if( ch <= 247 && i+3 < len && (str[i+1] & 0xc0) == 0x80 && (str[i+2] & 0xc0) == 0x80 && (str[i+3] & 0xc0) == 0x80) {
            int val = (int)(((ch & 15) << 18) | ((str[i+1] & 63) << 12) | ((str[i+2] & 63) << 6) | (str[i+3] & 63));
            if( val > 1114111 ) val = 65533;
            charcode = val;
            i += 3;
        }
        else
        {
            charcode = 65533;
            while(i+1 < len && (str[i+1] & 0xc0) == 0x80)
                i++;
        }
        u32buf.push_back(charcode);
    }

    // step 2. form segments, for each segment find the proper font, direction and script.
    len = (int)u32buf.size();
    unsigned* chars = &u32buf[0];
    int prev_dy = 0;

#ifdef HAVE_HARFBUZZ
    hb_direction_t glob_dir = HB_DIRECTION_INVALID;
#endif

    while(len > 0)
    {
        int nextline_dy = 0;
        font_t* ttface0 = fontface->ttface;
        float scale0 = fontface->scale;
        font_t* curr_ttface = ttface0;

        segments.clear();
        glyphs.clear();

    #ifdef HAVE_HARFBUZZ
        hb_script_t curr_script = HB_SCRIPT_UNKNOWN;
        hb_direction_t curr_dir = HB_DIRECTION_INVALID;
        int curr_fontidx = -1;

        int k, segstart = 0, punctstart = -1;
        // TODO: possibly implement https://unicode.org/reports/tr9/ or find compact implementation of it
        for(i = 0; i < len; i++)
        {
            int c = chars[i];
            if(c == '\n')
                break;
            hb_unicode_general_category_t cat = hb_unicode_general_category(hb_uni_funcs, c);
            hb_script_t script = hb_unicode_script(hb_uni_funcs, c);
            int glyph_index = stbtt_FindGlyphIndex(curr_ttface, c);
            bool is_rtl = isRightToLeft(c);
            hb_direction_t dir = HB_DIRECTION_INVALID;

            if(script == HB_SCRIPT_ARABIC || script == HB_SCRIPT_HEBREW ||
               script == HB_SCRIPT_SYRIAC || script == HB_SCRIPT_THAANA || is_rtl)
                dir = HB_DIRECTION_RTL;
            else if(script == HB_SCRIPT_LATIN || script == HB_SCRIPT_CYRILLIC || script == HB_SCRIPT_GREEK ||
               (cat >= HB_UNICODE_GENERAL_CATEGORY_SURROGATE &&
                cat <= HB_UNICODE_GENERAL_CATEGORY_CONNECT_PUNCTUATION) ||
                cat == HB_UNICODE_GENERAL_CATEGORY_CURRENCY_SYMBOL)
                dir = HB_DIRECTION_LTR;

            bool is_punct = dir == HB_DIRECTION_INVALID &&
                ((c < 128 && (ispunct((char)c) || isspace((char)c))) ||
               (cat >= HB_UNICODE_GENERAL_CATEGORY_DASH_PUNCTUATION &&
                cat <= HB_UNICODE_GENERAL_CATEGORY_OPEN_PUNCTUATION) ||
               (cat >= HB_UNICODE_GENERAL_CATEGORY_MODIFIER_SYMBOL &&
                cat <= HB_UNICODE_GENERAL_CATEGORY_SPACE_SEPARATOR) ||
                 cat == HB_UNICODE_GENERAL_CATEGORY_FORMAT);

            // when we switch to fallback (e.g. uni) font for some characters after quote or open paren,
            // and when we meet the corresponding closing quote/closing paren, we would prefer to
            // take it from the same font. this hack helps to achieve this effect in some cases
            if (is_punct && c < 256 && cat != HB_UNICODE_GENERAL_CATEGORY_FORMAT &&
                cat != HB_UNICODE_GENERAL_CATEGORY_SPACE_SEPARATOR && !isspace((char)c) &&
                curr_script != HB_SCRIPT_LATIN && curr_script != HB_SCRIPT_CYRILLIC && curr_dir != HB_DIRECTION_RTL)
            {
                is_punct = false;
                script = HB_SCRIPT_LATIN;
                dir = HB_DIRECTION_LTR;
            }

        #if 0
            printf("%d. c=%d, scr=%c%c%c%c, cat=%d, is_rtl=%d, dir=%s, is_punct=%d\n", i, c,
            (script>>24)&255, (script>>16)&255, (script>>8)&255, (script>>0)&255, (int)cat, (int)is_rtl,
             (dir == HB_DIRECTION_INVALID? "invalid" : dir == HB_DIRECTION_LTR ? "left" : "right"), (int)is_punct);
        #endif

            if(glyph_index != 0 ||
               (cat == HB_UNICODE_GENERAL_CATEGORY_FORMAT &&
                dir == HB_DIRECTION_INVALID))
            {
                if(is_punct)
                {
                    if(punctstart < 0)
                        punctstart = i;
                    continue;
                }

                if(cat == HB_UNICODE_GENERAL_CATEGORY_SURROGATE || cat == HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK ||
                   ((script == curr_script || curr_script == HB_SCRIPT_UNKNOWN || curr_script == HB_SCRIPT_COMMON ||
                    script == HB_SCRIPT_UNKNOWN || script == HB_SCRIPT_COMMON || script == HB_SCRIPT_INHERITED) &&
                    (dir == curr_dir || curr_dir == HB_DIRECTION_INVALID || dir == HB_DIRECTION_INVALID)))
                {
                    punctstart = -1;
                    if(curr_dir == HB_DIRECTION_INVALID)
                        curr_dir = dir;
                    if(curr_script == HB_SCRIPT_UNKNOWN || curr_script == HB_SCRIPT_INHERITED)
                        curr_script = script;
                    continue;
                }
            }

            if(glob_dir == HB_DIRECTION_INVALID)
            {
                glob_dir = curr_dir;
                if(glob_dir == HB_DIRECTION_INVALID)
                    glob_dir = dir;
            }

            font_t* ttface = 0;
            for(j = -1; j < BUILTIN_FONTS_NUM; j++)
            {
                ttface = j < 0 ? ttface0 : builtin_ffaces[j]->ttface;
                glyph_index = stbtt_FindGlyphIndex(ttface, c);
                if(glyph_index != 0)
                    break;
                if(j+1 == BUILTIN_FONTS_NUM)
                {
                    chars[i] = c = '?'; // replace the character with 'unknown' ~ <?> (TBD: replace ? with 0xFFFD)
                    break;
                }
            }
            int fontidx = j;
            int seglen = i - segstart;
            // if the current segment (if any) ends with 1 or more neutral/punctuation characters,
            // we have 3 choices:
            //   1. attach this "tail" to the current segment
            //   2. make this "tail" the head of the new segment
            //   3. split those characters somehow between the current and the new segment
            // if the current segment direction matches the global direction, we choose the option 1.
            // otherwise we choose the option 3, depending on which characters from the tail
            // are available in the new font for the new segment.
            // Unless they are some specific marks, they are all available, in which case
            // the option 3 turns into the option 2.
            if(seglen > 0 && punctstart >= 0 && glob_dir != curr_dir)
            {
                for(k = i; k > punctstart; k--)
                {
                    if(stbtt_FindGlyphIndex(ttface, chars[k-1]) == 0)
                        break;
                }
                seglen = k - segstart;
            }

            if(seglen > 0)
            {
                //printf("segment of %d characters pushed\n", seglen);
                segments.push_back(TextSegment(curr_ttface, curr_fontidx,
                    segstart, segstart + seglen, curr_script, curr_dir));
            }
            segstart += seglen;
            punctstart = punctstart >= 0 && is_punct ? segstart : -1;
            curr_ttface = ttface;
            curr_fontidx = fontidx;
            curr_script = script;
            curr_dir = dir;
        }

        if(i > segstart)
        {
            segments.push_back(TextSegment(curr_ttface, curr_fontidx,
                segstart, i, curr_script, curr_dir));
        }

        if(glob_dir == HB_DIRECTION_INVALID)
            glob_dir = HB_DIRECTION_LTR;

        int nsegments = (int)segments.size();
        ttface0 = segments[0].ttface;

        if(nsegments > 0)
        {
            TextSegment& seg = segments[0];
            if(seg.dir == HB_DIRECTION_INVALID)
                seg.dir = glob_dir;
            if(seg.script == HB_SCRIPT_INVALID)
                seg.script = HB_SCRIPT_COMMON;
        }

        // step 3. try to merge some segments
        for(k = 0, j = 1; j < nsegments; j++)
        {
            TextSegment& seg = segments[j];
            if(seg.dir == HB_DIRECTION_INVALID)
                seg.dir = glob_dir;
            hb_script_t script = seg.script;
            if(script == HB_SCRIPT_INVALID)
                seg.script = script = HB_SCRIPT_COMMON;
            TextSegment& prev = segments[k];
            if(seg.ttface == prev.ttface && seg.dir == prev.dir &&
               (script == prev.script ||
                ((script == HB_SCRIPT_COMMON || script == HB_SCRIPT_LATIN ||
                  script == HB_SCRIPT_CYRILLIC || script == HB_SCRIPT_GREEK ||
                  script == HB_SCRIPT_HAN || script == HB_SCRIPT_HIRAGANA) &&
                 (prev.script == HB_SCRIPT_COMMON || prev.script == HB_SCRIPT_LATIN ||
                  prev.script == HB_SCRIPT_CYRILLIC || prev.script == HB_SCRIPT_GREEK ||
                  prev.script == HB_SCRIPT_HAN || prev.script == HB_SCRIPT_HIRAGANA))))
            {
                prev.end = seg.end;
                if(prev.script != script)
                    prev.script = HB_SCRIPT_COMMON;
            }
            else
            {
                k++;
                if(j > k)
                    segments[k] = seg;
            }
        }

        if(nsegments > 0)
            segments.resize(k+1);
        else
            nextline_dy = prev_dy;
        nsegments = (int)segments.size();

        if(glob_dir == HB_DIRECTION_RTL)
        {
            std::reverse(segments.begin(), segments.end());
        }
        //printf("%s: nsegments=%d\n", str_.c_str(), nsegments);

        // step 4. shape each text segment using Harfbuzz
        for(j = 0; j < nsegments; j++)
        {
            const TextSegment& seg = segments[j];
            int fontidx = seg.fontidx;
            FontFace& fface = fontidx < 0 ? fontface : builtin_ffaces[fontidx];
            font_t* ttface = fface->ttface;
            hb_font_t* hb_font = fface->hb_font;
            hb_buffer_reset(hb_buf);
            hb_buffer_add_utf32(hb_buf, chars, len, seg.start, seg.end - seg.start);
            hb_buffer_set_direction(hb_buf, seg.dir);
            hb_buffer_set_script(hb_buf, seg.script);
            hb_shape(hb_font, hb_buf, 0, 0);

            unsigned nglyphs = 0;
            hb_glyph_info_t *ginfo = hb_buffer_get_glyph_infos(hb_buf, &nglyphs);
            hb_glyph_position_t* gpos = hb_buffer_get_glyph_positions(hb_buf, &nglyphs);

            for(k = 0; k < (int)nglyphs; k++)
            {
                FontGlyph glyph(ttface, ginfo[k].codepoint, gpos[k], fface->scale);
                glyphs.push_back(glyph);
            }
        }
    #else
        for(i = 0; i < len; i++)
        {
            int c = chars[i];
            if(c == '\n')
                break;
            font_t* ttface = ttface0;
            float scale = scale0;
            int q_glyph_index = stbtt_FindGlyphIndex(ttface0, '?');
            for(j = -1; j < BUILTIN_FONTS_NUM; j++)
            {
                if (j >= 0)
                {
                    ttface = builtin_ffaces[j]->ttface;
                    scale = builtin_ffaces[j]->scale;
                }
                int glyph_index = ttface ? stbtt_FindGlyphIndex(ttface, c) : 0;
                if(glyph_index == 0)
                {
                    if (j+1 < BUILTIN_FONTS_NUM)
                        continue;
                    ttface = ttface0;
                    scale = scale0;
                    glyph_index = q_glyph_index;
                }
                hb_glyph_position_t pos;
                pos.x_advance = pos.y_advance = 0;
                pos.x_offset = pos.y_offset = 0;
                glyphs.push_back(FontGlyph(ttface, glyph_index, pos, scale));
                break;
            }
        }
        if (i == 0)
            nextline_dy = prev_dy;
    #endif

        chars += i;
        len -= i;

        if(len > 0 && chars[0] == '\n')
        {
            chars++;
            len--;
        }

        // step 5. finally, let's render it
        // (TODO: in the case of right alignment we need to render it from the right-most character)
        int nglyphs = (int)glyphs.size();
        int space_glyph = -1;
        curr_ttface = 0;
        min_x = std::min(min_x, pen.x);
        max_x = std::max(max_x, pen.x);

        curr_ttface = 0;
        int ascent = 0, descent = 0, linegap = 0;
        for(j = 0; j < nglyphs; j++)
        {
            const FontGlyph& glyph = glyphs[alignment == PUT_TEXT_ALIGN_RIGHT ? nglyphs - j - 1 : j];
            font_t* ttface = glyph.ttface;
            if(ttface != curr_ttface)
            {
                curr_ttface = ttface;
                space_glyph = stbtt_FindGlyphIndex(ttface, ' ');
                stbtt_GetFontVMetrics(ttface, &ascent, &descent, &linegap);
                if (linegap == 0) linegap = ascent - descent;
            }

            float scale = glyph.scale;
            GlyphCacheKey key(curr_ttface, glyph.index, size, weight, scale);
            GlyphCacheVal* cached = findCachedGlyph(key);
            const uchar* bitmap_buf = 0;
            int bitmap_step = 0;
            Rect bbox;

            if(!cached)
            {
                cached = &new_cached;
                int w=0, h=0, xoff=0, yoff=0;
                float advx = 0.f;
                bitmap_buf = stbtt_GetGlyphBitmapSubpixelRealloc(ttface, scale, scale, 0.f, 0.f,
                                glyph.index, &w, &h, &bitmap_step, &xoff, &yoff, &advx, &glyph_buf, &glyph_bufsz);
                if(!bitmap_buf)
                    continue;
                //printf("j=%d. bw=%d, bh=%d, step=%d, xoff=%d, yoff=%d, advx=%.1f, glyph_bufsz=%d\n", j, w, h, bitmap_step, xoff, yoff, advx, glyph_bufsz);

                cached->width = w;
                cached->height = h;
                cached->advance.x = cvRound(advx*64);
                cached->advance.y = 0;
                cached->ascent = cvRound(ascent*glyph.scale);
                cached->linegap = cvRound(linegap*glyph.scale);
                cached->horiBearingX = (int)(xoff*64);
                cached->horiBearingY = (int)(-yoff*64);

                bbox = Rect(0, 0, w, h);

                if(w <= MAX_CACHED_GLYPH_SIZE && h <= MAX_CACHED_GLYPH_SIZE)
                {
                    compressCharacter(bitmap_buf, bitmap_step, Size(w, h),
                                      cached->rlebuf, cached->bbox,
                                      0 //&cached->crc
                                      );
                    addToCache(key, *cached);
                }
            }
            else
            {
                if(render)
                {
                    CV_Assert(decompressCharacter(cached->rlebuf, cached->bbox,
                                                  pixbuf, 0 //&cached->crc
                                                  ));
                }
                bitmap_buf = pixbuf.empty() ? 0 : &pixbuf[0];
                bitmap_step = cached->bbox.width;
                bbox = cached->bbox;
            }

            const hb_glyph_position_t& pos = glyph.pos;
            int dx = cached->advance.x >> 6;
            int dy = cached->advance.y >> 6;
            int new_pen_x = pen.x + dx*alignSign;
            nextline_dy = max(nextline_dy, cached->linegap);
            // TODO: this wrapping algorithm is quite dumb,
            // preferably should split text at word boundary
            if( wrap && imgsize.width > 0 && new_pen_x*alignSign > x1*alignSign )
            {
                pen.y += nextline_dy;
                dy = 0;
                pen.x = x0;
                wrapped = true;
                //max_baseline = 0;
                if(glyph.index == space_glyph) continue;
                new_pen_x = pen.x + dx*alignSign;
            }

            if(!wrapped)
                max_dy = std::max(max_dy, cached->ascent);
            int baseline = cached->height - (cached->horiBearingY >> 6);
            max_baseline = std::max(max_baseline, baseline);

            if(alignment == PUT_TEXT_ALIGN_RIGHT)
                pen.x = new_pen_x;

            int x = pen.x + bbox.x + ((pos.x_offset + cached->horiBearingX) >> 6);
            int y = pen.y + (bottom_left ? 1 : -1)*(((pos.y_offset + cached->horiBearingY) >> 6) - bbox.y);

            if( render )
            {
                //circle(img, pen, 2, Scalar(0, 0, 128), -1, LINE_AA);
                drawCharacter(img, color, bitmap_buf, bitmap_step, bbox.size(), x, y, bottom_left);
            }

            pen.x = new_pen_x;
            pen.y += dy;
            min_x = std::min(min_x, pen.x);
            max_x = std::max(max_x, pen.x);
        }

        if(len > 0 && (pen.x != x0 || segments.empty()))
        {
            pen.x = x0;
            pen.y += nextline_dy;
            prev_dy = nextline_dy;
        }
    }

    if(bbox_)
    {
        if(flags & PUT_TEXT_ORIGIN_BL)
            *bbox_ = Rect(min_x, org.y - max_baseline, max_x - min_x + 1, pen.y - org.y + max_dy + max_baseline);
        else
            *bbox_ = Rect(min_x, org.y - max_dy, max_x - min_x + 1, pen.y - org.y + max_dy + max_baseline);
    }

    // restore the weights
    if (weight != 0)
        for(j = 0; j <= BUILTIN_FONTS_NUM; j++)
        {
            font_t* ttface = (j < BUILTIN_FONTS_NUM ? builtin_ffaces[j] : fontface)->ttface;
            if (!ttface || stbtt_GetWeight(ttface) == saved_weights[j])
                continue;
            int params[] = {STBTT_FOURCC('w', 'g', 'h', 't'), saved_weights[j]};
            stbtt_SetInstance(ttface, params, 1, 0);
        }

    return pen;
}


Point putText(InputOutputArray img_, const String& str,
             Point org, Scalar color_,
             FontFace& fontface, int size,
             int weight, PutTextFlags flags, Range wrap)
{
    uchar color[] =
    {
        saturate_cast<uchar>(color_[0]), saturate_cast<uchar>(color_[1]),
        saturate_cast<uchar>(color_[2]), saturate_cast<uchar>(color_[3])
    };
    Mat img = img_.getMat();
    int nch = img.channels();
    CV_Assert(img.depth() == CV_8U);
    CV_Assert(nch == 1 || nch == 3 || nch == 4);

    return fontRenderEngine.putText_(img, img.size(), str, org, color, fontface, size,
                                     weight, flags, wrap, true, 0);
}


Rect getTextSize(Size imgsize, const String& str, Point org,
                 FontFace& fontface, int size,
                 int weight, PutTextFlags flags, Range wrap)
{
    Mat img;
    Rect bbox;
    fontRenderEngine.putText_(img, imgsize, str, org, 0, fontface, size,
                              weight, flags, wrap, false, &bbox);
    return bbox;
}

}

//////////////////////////// text drawing functions for backward compatibility ///////////////////////////

namespace cv
{

static void hersheyToTruetype(int fontFace, double fontScale, int thickness,
                              String& ttname, int& ttsize, int& ttweight)
{
    double sf = 0;
    switch(fontFace & ~FONT_ITALIC)
    {
    case FONT_HERSHEY_PLAIN:
        ttname = "sans";
        sf = 6.6;
        ttweight = thickness <= 1 ? 400 : 800;
        break;
    case FONT_HERSHEY_SIMPLEX:
        ttname = "sans";
        sf = 3.7;
        ttweight = thickness <= 1 ? 400 : 600;
        break;
    case FONT_HERSHEY_DUPLEX:
        ttname = "sans";
        sf = 3.7;
        ttweight = thickness <= 1 ? 600 : 800;
        break;
    case FONT_HERSHEY_COMPLEX:
        ttname = "serif";
        sf = 3.7;
        ttweight = thickness <= 1 ? 400 : 800;
        break;
    case FONT_HERSHEY_TRIPLEX:
        ttname = "serif";
        sf = 3.7;
        ttweight = thickness <= 1 ? 600 : 800;
        break;
    case FONT_HERSHEY_COMPLEX_SMALL:
        ttname = "serif";
        sf = 4.6;
        ttweight = thickness <= 1 ? 400 : 800;
        break;
    case FONT_HERSHEY_SCRIPT_COMPLEX:
        ttname = "italic";
        sf = 4.0;
        ttweight = thickness <= 1 ? 400 : 600;
        break;
    case FONT_HERSHEY_SCRIPT_SIMPLEX:
        ttname = "italic";
        sf = 4.0;
        ttweight = thickness <= 1 ? 300 : 500;
        break;
    default:
        CV_Error(Error::StsBadArg, "Unknown font");
    }

    ttsize = cvRound(100*fontScale/sf);
}

void putText( InputOutputArray _img, const String& text, Point org,
              int fontFace, double fontScale, Scalar color,
              int thickness, int, bool bottomLeftOrigin )

{
    String ttname;
    int ttsize = 0;
    int ttweight = 0;
    hersheyToTruetype(fontFace, fontScale, thickness, ttname, ttsize, ttweight);
    FontFace fface(ttname);
    PutTextFlags flags = bottomLeftOrigin ? PUT_TEXT_ORIGIN_BL : PUT_TEXT_ORIGIN_TL;
    putText(_img, text, org, color, fface, ttsize, ttweight, flags);
}

Size getTextSize(const String& text, int fontFace, double fontScale, int thickness, int* _base_line)
{
    String ttname;
    int ttsize = 0;
    int ttweight = 0;
    hersheyToTruetype(fontFace, fontScale, thickness, ttname, ttsize, ttweight);

    FontFace fface(ttname);
    Rect r = getTextSize(Size(), text, Point(), fface, ttsize, ttweight, PUT_TEXT_ALIGN_LEFT);

    int baseline = r.y + r.height;
    if(_base_line)
        *_base_line = baseline;
    return Size(r.width, r.height - baseline);
}

double getFontScaleFromHeight(const int fontFace, const int pixelHeight, const int thickness)
{
    String ttname;
    int ttsize = 0;
    int ttweight = 0;
    hersheyToTruetype(fontFace, 1.0, thickness, ttname, ttsize, ttweight);
    return (double)pixelHeight/ttsize;
}

}
