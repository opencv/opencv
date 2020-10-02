// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#if defined HAVE_FREETYPE && defined HAVE_HARFBUZZ

#include <list>
#include <tuple>
#include <unordered_map>

#include "zlib.h"
#include "ft2build.h"
#include FT_FREETYPE_H
#include FT_MULTIPLE_MASTERS_H

#include "hb-ft.h"

namespace cv
{

#include "builtin_font0.h"
#include "builtin_font1.h"
#include "builtin_font2.h"

typedef struct BuiltinFontData
{
    const uchar* gzdata;
    size_t size;
    const char* name;
    double sf;
    bool italic;
    int dweight;
} BuiltinFontData;

enum
{
    BUILTIN_FONTS_NUM = 3
};

static BuiltinFontData builtinFontData[BUILTIN_FONTS_NUM+1] =
{
    {OcvBuiltinFontSans, sizeof(OcvBuiltinFontSans), "sans", 1.0, false, 0},
    {OcvBuiltinFontItalic, sizeof(OcvBuiltinFontItalic), "italic", 1.0, true, 0},
    {OcvBuiltinFontUni, sizeof(OcvBuiltinFontUni), "uni", 1.0, false, 0},
    {0, 0, 0, 0.0, false, 0}
};

struct FreeTypeLib
{
    FreeTypeLib() { library = 0; }
    ~FreeTypeLib()
    {

    }

    FT_Library  library;   /* handle to library     */
};

static bool inflate(const void* src, size_t srclen, std::vector<uchar>& dst)
{
    dst.resize((size_t)(srclen*2.5));
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
        ftface = 0;
        scalefactor = 1.0;
        italic = false;
        dweight = 0;
        hb_font = 0;
    }

    ~Impl()
    {
        deleteFont();
    }

    void deleteFont()
    {
        hb_font_destroy(hb_font);
        hb_font = 0;
        if(ftface != 0)
            FT_Done_Face(ftface);
        ftface = 0;
        currname.clear();
        initParams();
    }

    void initParams()
    {
        currweight = -1;
        currsize = -1;
        currflags = 0;
    }

    bool setStd(FT_Library library, const BuiltinFontData& fontdata)
    {
        if(ftface == 0 || currname != fontdata.name)
        {
            deleteFont();
            if(!inflate(fontdata.gzdata, fontdata.size, fontbuf))
                return false;
            int err = FT_New_Memory_Face(library, &fontbuf[0], (FT_Long)fontbuf.size(), 0, &ftface);
            if(err != 0)
                return false;
        }
        currname = fontdata.name;
        scalefactor = fontdata.sf;
        italic = fontdata.italic;
        dweight = fontdata.dweight;
        initParams();
        return true;
    }

    bool set(FT_Library library, const String& fontname, double sf)
    {
        CV_Assert(!fontname.empty());

        if(ftface != 0 && fontname == currname)
        {
            scalefactor = sf;
            return true;
        }

        deleteFont();

        int err = FT_New_Face(library, fontname.c_str(), 0, &ftface);
        if(err != 0) { ftface = 0; return false; }
        currname = fontname;
        scalefactor = sf;
        initParams();
        return true;
    }

    bool setParams(FT_Library library, double size0, int weight, int flags)
    {
        flags &= PUT_TEXT_SIZE_MASK;
        int sizeunits = flags & PUT_TEXT_SIZE_MASK;
        int size = cvRound(size0*scalefactor*(sizeunits == PUT_TEXT_SIZE_POINTS ? 64 : 1));

        if (ftface == 0)
            return false;
        if (std::abs(size - currsize) < 1e-3 &&
            weight == currweight &&
            flags == currflags)
            return true;

        FT_MM_Var* multimaster = 0;

        // retrieve the variable font information, if any
        if( weight != currweight &&
            FT_Get_MM_Var( ftface, &multimaster ) == 0 )
        {
            FT_Fixed design_pos[16];

            for ( int n = 0; n < (int)multimaster->num_axis; n++ )
            {
                design_pos[n] = multimaster->axis[n].def;
                FT_Fixed minval = multimaster->axis[n].minimum;
                FT_Fixed maxval = multimaster->axis[n].maximum;
                const char* name = multimaster->axis[n].name;
                if(strcmp(name, "Weight") == 0)
                {
                    FT_Fixed w = cvRound(((weight <= 0 ? 400 : weight) + dweight)*(1<<16));
                    w = w < minval ? minval : w > maxval ? maxval : w;
                    design_pos[n] = w;
                }
                else if(italic && strcmp(name, "Slant") == 0)
                {
                    design_pos[n] = minval;
                }
            }
            FT_Set_Var_Design_Coordinates(ftface, multimaster->num_axis, design_pos);
        }

        if(multimaster)
            FT_Done_MM_Var(library, multimaster);

        if(size != currsize || flags != currflags)
        {
            if(sizeunits == PUT_TEXT_SIZE_POINTS)
            {
                FT_Fixed sz = (FT_Fixed)size;
                FT_Set_Char_Size(ftface, sz, sz, 120, 120);
            }
            else
            {
                FT_Set_Pixel_Sizes(ftface, (FT_Fixed)size, (FT_Fixed)size);
            }
        }

        if(!hb_font)
            hb_font = hb_ft_font_create_referenced(ftface);
        else
            hb_ft_font_changed(hb_font);

        currweight = weight;
        currsize = size;
        currflags = flags;
        return true;
    }

    String currname;
    double scalefactor;
    bool italic;
    int dweight;
    int currsize;
    int currweight;
    int currflags;
    FT_Face ftface;
    hb_font_t* hb_font;
    std::vector<uchar> fontbuf;
};

struct GlyphCacheKey
{
    GlyphCacheKey()
    {
        ftface = 0;
        size = 0;
        weight = 0;
        flags = 0;
    }
    GlyphCacheKey(FT_Face ftface_, double size_, int weight_, int flags_)
    {
        ftface = ftface_;
        size = size_;
        weight = weight_;
        flags = flags_;
    }

    FT_Face ftface;
    int size;
    int weight;
    int flags;
};

static bool operator == (const GlyphCacheKey& k1, const GlyphCacheKey& k2)
{
    return k1.ftface == k2.ftface && k1.size == k2.size && k1.weight == k2.weight && k1.flags == k2.flags;
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
        size_t hs[] = {(size_t)(void*)key.ftface, (size_t)key.size, (size_t)key.weight, (size_t)key.flags};
        return hash_seq(hs, sizeof(hs)/sizeof(hs[0]));
    }
};

struct GlyphCacheVal
{
    GlyphCacheVal()
    {
    }

    GlyphCacheVal(Rect brect_)
    {
        brect = brect_;
    }

    bool empty()
    {
        return brect.area() == 0;
    }

    std::vector<uchar> rlebuf;
    Rect brect;
};

typedef std::pair<GlyphCacheKey, GlyphCacheVal> GlyphCacheEntry;
typedef std::list<GlyphCacheEntry>::iterator GlyphCacheEntryIt;

struct TextSegment
{
    TextSegment()
    {
        ftface = 0;
        fontidx = 0;
        start = end = 0;
        script = HB_SCRIPT_UNKNOWN;
        dir = HB_DIRECTION_LTR;
    }
    TextSegment(FT_Face ftface_, int fontidx_, int start_, int end_, hb_script_t script_, hb_direction_t dir_)
    {
        ftface = ftface_;
        fontidx = fontidx_;
        start = start_;
        end = end_;
        script = script_;
        dir = dir_;
    }
    FT_Face ftface;
    int fontidx;
    int start, end;
    hb_script_t script;
    hb_direction_t dir;
};

struct FontGlyph
{
    FontGlyph() { ftface = 0; index = -1; }
    FontGlyph(FT_Face ftface_, int glyph_index_, const hb_glyph_position_t& pos_)
    {
        ftface = ftface_;
        index = glyph_index_;
        pos = pos_;
    }
    FT_Face ftface;
    int index;
    hb_glyph_position_t pos;
};

class FontRenderEngine
{
public:
    FontRenderEngine()
    {
        ftlib = 0;
        hb_buf = hb_buffer_create();
        hb_buffer_guess_segment_properties(hb_buf);
        hb_uni_funcs = hb_unicode_funcs_get_default();
        max_cache_size = 1000;
    }

    ~FontRenderEngine()
    {
        hb_buffer_destroy(hb_buf);
        for(int i = 0; i < BUILTIN_FONTS_NUM; i++)
            builtin_ffaces[i] = FontFace();
        if(ftlib)
        {
            FT_Done_FreeType(ftlib);
            ftlib = 0;
        }
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

    const GlyphCacheVal& findCachedGlyph(const GlyphCacheKey& key)
    {
        auto it = glyph_cache.find(key);
        if(it == glyph_cache.end())
            return not_found;
        all_cached.splice(all_cached.begin(), all_cached, it->second);
        return (*it->second).second;
    }

    FontFace& getStdFontFace(int i)
    {
        CV_Assert(i >= 0 && i < BUILTIN_FONTS_NUM);
        builtin_ffaces[i]->setStd(getLibrary(), builtinFontData[i]);
        return builtin_ffaces[i];
    }

    FT_Library getLibrary()
    {
        if(!ftlib)
        {
            int err = FT_Init_FreeType( &ftlib );
            if(err != 0) { ftlib = 0; return 0; }
        }
        return ftlib;
    }

    Point putText_( Mat& img, Size imgsize, const String& str_, Point org,
                    const uchar* color, FontFace& fontface, double size,
                    int weight, int flags, Range wrapRange, bool render,
                    Rect* brect );

protected:
    FT_Library ftlib;
    FontFace builtin_ffaces[BUILTIN_FONTS_NUM];
    bool builtin_ffaces_initialized;

    hb_unicode_funcs_t* hb_uni_funcs;

    // LRU cache of glyphs
    GlyphCacheVal not_found;
    std::list<GlyphCacheEntry> all_cached;
    std::unordered_map<GlyphCacheKey, GlyphCacheEntryIt, GlyphCacheHash> glyph_cache;
    size_t max_cache_size;

    hb_buffer_t* hb_buf;
    std::vector<unsigned> u32buf;
    std::vector<TextSegment> segments;
    std::vector<FontGlyph> glyphs;
};

thread_local FontRenderEngine fontRenderEngine;

FontFace::FontFace() { impl = makePtr<Impl>(); }
FontFace::FontFace(const String& fontname, double sf_ )
{
    impl = makePtr<Impl>();
    set(fontname, sf_);
}

bool FontFace::set(const String& fontname_, double sf)
{
    String fontname = fontname_;
    if(fontname.empty())
        fontname = "sans";
    if(impl->ftface != 0 && impl->currname == fontname)
        return true;
    int i = 0;
    for( ; i < BUILTIN_FONTS_NUM; i++ )
    {
        if( builtinFontData[i].name == fontname )
            break;
    }
    if( i >= BUILTIN_FONTS_NUM )
        i = -1;

    FontRenderEngine& engine = fontRenderEngine;

    bool ok = false;
    if( i >= 0 )
    {
        FontFace& builtin_fface = engine.getStdFontFace(i);
        if(builtin_fface.impl->ftface)
        {
            impl = builtin_fface.impl;
            ok = true;
        }
    }
    else
    {
        sf = sf > 0 ? sf : 1.0;
        if(impl->ftface != 0)
            impl = makePtr<Impl>();
        ok = impl->set(engine.getLibrary(), fontname, sf);
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
        if(builtinFontData[i].name == fontname)
        {
            FontFace& builtin_fface = fontRenderEngine.getStdFontFace(i);
            if( builtin_fface.impl->ftface )
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
double FontFace::getScaleFactor() const { return impl->scalefactor; }
FontFace::Impl* FontFace::operator -> () { return impl.get(); }
FontFace::~FontFace() {}

static void drawCharacter(
    Mat& img, const uchar* color,
    const FT_Bitmap* bitmap, int x0, int y0,
    bool bottom_left )
{
    int nch = img.channels();
    int bw = (int)(bitmap->width), bh = (int)(bitmap->rows);
    int rows = img.rows, cols = img.cols;
    uchar b = color[0], g = color[1], r = color[2];
    const uchar* bitmap_buf = bitmap->buffer;
    int bitmap_pitch = bitmap->pitch;

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
            uchar alpha = bitmap_buf[dy*bitmap_pitch + dx];
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


Point FontRenderEngine::putText_(
    Mat& img, Size imgsize, const String& str_, Point org,
    const uchar* color, FontFace& fontface, double size,
    int weight, int flags, Range wrapRange, bool render, Rect* brect )
{
    FT_Library ftlib_ = getLibrary();
    int load_glyph_flag = render ? FT_LOAD_RENDER : FT_LOAD_DEFAULT;
    bool bottom_left = (flags & PUT_TEXT_ORIGIN_BL) != 0;

    if(fontface.getName().empty())
        fontface.set("sans");

    fontface->setParams(ftlib, size, weight, flags);
    if(!fontface->ftface)
        return org;

    for(int j = 0; j < BUILTIN_FONTS_NUM; j++)
    {
        if(!builtin_ffaces_initialized)
            builtin_ffaces[j].set(builtinFontData[j].name);
        builtin_ffaces[j]->setParams(ftlib_, size, weight, flags);
    }
    builtin_ffaces_initialized = true;

    Point pen = org;
    int i, j, k, len = (int)str_.size();
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
        if(brect)
            *brect = Rect(org.x, org.y, 0, 0);
        return org;
    }

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
    hb_direction_t glob_dir = HB_DIRECTION_INVALID;

    while(len > 0)
    {
        int nextline_dy = 0;
        int segstart = 0, punctstart = -1;
        FT_Face ftface0 = fontface->ftface;
        FT_Face curr_ftface = ftface0;
        int curr_fontidx = -1;
        hb_script_t curr_script = HB_SCRIPT_UNKNOWN;
        hb_direction_t curr_dir = HB_DIRECTION_INVALID;

        segments.clear();
        glyphs.clear();

        // TODO: possibly implement https://unicode.org/reports/tr9/ or find compact implementation of it
        for(i = 0; i < len; i++)
        {
            unsigned c = chars[i];
            if(c == '\n')
                break;
            hb_unicode_general_category_t cat = hb_unicode_general_category(hb_uni_funcs, c);
            hb_script_t script = hb_unicode_script(hb_uni_funcs, c);
            FT_UInt glyph_index = FT_Get_Char_Index(curr_ftface, c);
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
               ((cat == HB_UNICODE_GENERAL_CATEGORY_FORMAT/* ||
                 cat == HB_UNICODE_GENERAL_CATEGORY_CONTROL*/) &&
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

            FT_Face ftface = 0;
            for(j = -1; j < BUILTIN_FONTS_NUM; j++)
            {
                ftface = j < 0 ? ftface0 : builtin_ffaces[j]->ftface;
                glyph_index = FT_Get_Char_Index(ftface, c);
                if(glyph_index != 0)
                    break;
                if(j+1 == BUILTIN_FONTS_NUM)
                {
                    chars[i] = c = 0xFFFD; // replace the character with 'unknown' ~ <?>
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
                    if(FT_Get_Char_Index(ftface, chars[k-1]) == 0)
                        break;
                }
                seglen = k - segstart;
            }

            if(seglen > 0)
            {
                //printf("segment of %d characters pushed\n", seglen);
                segments.push_back(TextSegment(curr_ftface, curr_fontidx,
                    segstart, segstart + seglen, curr_script, curr_dir));
            }
            segstart += seglen;
            punctstart = punctstart >= 0 && is_punct ? segstart : -1;
            curr_ftface = ftface;
            curr_fontidx = fontidx;
            curr_script = script;
            curr_dir = dir;
        }

        if(i > segstart)
        {
            segments.push_back(TextSegment(curr_ftface, curr_fontidx,
                segstart, i, curr_script, curr_dir));
        }

        if(glob_dir == HB_DIRECTION_INVALID)
            glob_dir = HB_DIRECTION_LTR;

        int nsegments = (int)segments.size();
        ftface0 = segments[0].ftface;

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
            if(seg.ftface == prev.ftface && seg.dir == prev.dir &&
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
            FT_Face ftface = fface->ftface;
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
                FontGlyph glyph(ftface, ginfo[k].codepoint, gpos[k]);
                glyphs.push_back(glyph);
            }
        }

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
        curr_ftface = 0;
        min_x = std::min(min_x, pen.x);
        max_x = std::max(max_x, pen.x);

        for(j = 0; j < nglyphs; j++)
        {
            const FontGlyph& glyph = glyphs[alignment == PUT_TEXT_ALIGN_RIGHT ? nglyphs - j - 1 : j];
            FT_Face ftface = glyph.ftface;
            if(ftface != curr_ftface)
            {
                curr_ftface = ftface;
                space_glyph = (int)FT_Get_Char_Index(ftface, ' ');
            }
            int err = FT_Load_Glyph(ftface, (FT_UInt)glyph.index, load_glyph_flag);
            if(err != 0)
                continue;
            FT_GlyphSlot slot = glyph.ftface->glyph;
            const hb_glyph_position_t& pos = glyph.pos;
            int dx = (int)(slot->advance.x >> 6);
            int dy = (int)(slot->advance.y >> 6);
            int new_pen_x = pen.x + dx*alignSign;
            nextline_dy = max(nextline_dy, (int)(slot->metrics.vertAdvance >> 6));
            // TODO: this wrapping algorithm is quite dumb
            if( wrap && imgsize.width > 0 && new_pen_x*alignSign > x1*alignSign )
            {
                pen.y += nextline_dy;
                dy = 0;
                pen.x = x0;
                wrapped = true;
                max_baseline = slot->bitmap_top;
                if(glyph.index == space_glyph) continue;
                new_pen_x = pen.x + dx*alignSign;
            }

            if(!wrapped)
                max_dy = std::max(max_dy, slot->bitmap_top);
            int baseline = (slot->metrics.height - slot->metrics.horiBearingY) >> 6;
            max_baseline = std::max(max_baseline, baseline);

            if(alignment == PUT_TEXT_ALIGN_RIGHT)
                pen.x = new_pen_x;

            int x = pen.x + ((pos.x_offset + slot->metrics.horiBearingX) >> 6);
            int y = pen.y - ((pos.y_offset + slot->metrics.horiBearingY) >> 6);

            if( render )
                drawCharacter( img, color, &slot->bitmap, x, y, bottom_left );

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

    if(brect)
    {
        if(flags & PUT_TEXT_ORIGIN_BL)
            *brect = Rect(min_x, org.y - max_baseline, max_x - min_x + 1, pen.y - org.y + max_dy + max_baseline);
        else
            *brect = Rect(min_x, org.y - max_dy, max_x - min_x + 1, pen.y - org.y + max_dy + max_baseline);
    }

    return pen;
}


Point putText(InputOutputArray img_, const String& str,
             Point org, Scalar color_,
             FontFace& fontface, double size,
             int weight, int flags, Range wrap)
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
                 FontFace& fontface, double size,
                 int weight, int flags, Range wrap)
{
    Mat img;
    Rect brect;
    fontRenderEngine.putText_(img, imgsize, str, org, 0, fontface, size,
                              weight, flags, wrap, false, &brect);
    return brect;
}

}
#else
namespace cv
{

struct Impl {
public:
    Impl() {}
};

FontFace::FontFace() {}
FontFace::FontFace(const String&, double) {}

bool FontFace::set(const String&, double) { return false; }
String FontFace::getName() const { return String(); }
double FontFace::getScaleFactor() const { return 1.0; }
bool FontFace::getBuiltinFontData(const String&,
                                  const uchar*& data, size_t& size)
{
    data = 0;
    size = 0;
    return false;
}

FontFace::~FontFace() {}
FontFace::Impl* FontFace::operator -> () { return impl.get(); }

Point putText(InputOutputArray, const String&, Point org, Scalar,
             FontFace&, double, int, int, Range)
{
    CV_Error(Error::StsNotImplemented, "putText needs freetype2; recompile OpenCV with freetype2 enabled.");
    return org;
}

Rect getTextSize(Size, const String&, Point, FontFace&, double, int, int, Range)
{
    CV_Error(Error::StsNotImplemented, "putText needs freetype2; recompile OpenCV with freetype2 enabled.");
    return Rect();
}

}
#endif

//////////////////////////// text drawing functions for backward compatibility ///////////////////////////

namespace cv
{

static void hersheyToTruetype(int fontFace, double fontScale, int thickness,
                              String& ttname, double& ttsize, int& ttweight)
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
        sf = 3.4;
        ttweight = thickness <= 1 ? 400 : 600;
        break;
    case FONT_HERSHEY_DUPLEX:
        ttname = "sans";
        sf = 3.4;
        ttweight = thickness <= 1 ? 600 : 800;
        break;
    case FONT_HERSHEY_COMPLEX:
        ttname = "serif";
        sf = 3.4;
        ttweight = thickness <= 1 ? 400 : 800;
        break;
    case FONT_HERSHEY_TRIPLEX:
        ttname = "serif";
        sf = 3.4;
        ttweight = thickness <= 1 ? 600 : 800;
        break;
    case FONT_HERSHEY_COMPLEX_SMALL:
        ttname = "serif";
        sf = 4.6;
        ttweight = thickness <= 1 ? 400 : 800;
        break;
    case FONT_HERSHEY_SCRIPT_COMPLEX:
        ttname = "italic";
        sf = 3.9;
        ttweight = thickness <= 1 ? 400 : 600;
        break;
    case FONT_HERSHEY_SCRIPT_SIMPLEX:
        ttname = "italic";
        sf = 3.9;
        ttweight = thickness <= 1 ? 300 : 500;
        break;
    default:
        CV_Error(Error::StsBadArg, "Unknown font");
    }

    ttsize = 100*fontScale/sf;
}

void putText( InputOutputArray _img, const String& text, Point org,
              int fontFace, double fontScale, Scalar color,
              int thickness, int, bool bottomLeftOrigin )

{
    String ttname;
    double ttsize = 0;
    int ttweight = 0;
    hersheyToTruetype(fontFace, fontScale, thickness, ttname, ttsize, ttweight);
    FontFace fface(ttname);
    int flags = bottomLeftOrigin ? PUT_TEXT_ORIGIN_BL : PUT_TEXT_ORIGIN_TL;
    putText(_img, text, org, color, fface, ttsize, ttweight, flags);
}

Size getTextSize(const String& text, int fontFace, double fontScale, int thickness, int* _base_line)
{
    String ttname;
    double ttsize = 0;
    int ttweight = 0;
    hersheyToTruetype(fontFace, fontScale, thickness, ttname, ttsize, ttweight);

    FontFace fface(ttname);
    Rect r = getTextSize(Size(), text, Point(), fface, ttsize, ttweight, 0);

    int baseline = r.y + r.height;
    if(_base_line)
        *_base_line = baseline;
    return Size(r.width, r.height - baseline);
}

double getFontScaleFromHeight(const int fontFace, const int pixelHeight, const int thickness)
{
    String ttname;
    double ttsize = 0;
    int ttweight = 0;
    hersheyToTruetype(fontFace, 1.0, thickness, ttname, ttsize, ttweight);
    return pixelHeight/ttsize;
}

}
