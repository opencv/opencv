// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#ifdef HAVE_FREETYPE

#include "zlib.h"

#include "ft2build.h"
#include FT_FREETYPE_H
#include FT_MULTIPLE_MASTERS_H

namespace cv
{

#include "default_font0.h"
#include "default_font1.h"
#include "default_font2.h"
#include "default_font3.h"

typedef struct DefaultFontData
{
    const uchar* gzdata;
    size_t size;
    const char* name;
    double sf;
} DefaultFontData;

enum
{
    DEFAULT_FONTS_NUM = 4
};

static DefaultFontData defaultFontData[DEFAULT_FONTS_NUM+1] =
{
    {OcvDefaultFontSans, sizeof(OcvDefaultFontSans), "sans", 1.0},
    {OcvDefaultFontSerif, sizeof(OcvDefaultFontSerif), "serif", 1.0},
    {OcvDefaultFontItalic, sizeof(OcvDefaultFontItalic), "italic", 1.3},
    {OcvDefaultFontUni, sizeof(OcvDefaultFontUni), "uni", 1.05},
    {0, 0, 0}
};

struct FreeTypeLib
{
    FreeTypeLib() { library = 0; }
    ~FreeTypeLib()
    {
        if(library)
        {
            FT_Done_FreeType(library);
            library = 0;
        }
    }
    FT_Library getLibrary()
    {
        if(!library)
        {
            int err = FT_Init_FreeType( &library );
            if(err != 0) { library = 0; return 0; }
        }
        return library;
    }
    FT_Library  library;   /* handle to library     */
};

static thread_local FreeTypeLib ftlib;
static thread_local FontFace default_ffaces[DEFAULT_FONTS_NUM];

static bool inflate(const void* src, size_t srclen, std::vector<uchar>& dst)
{
    dst.resize((size_t)(srclen*2.5));
    for(int attempts = 0; attempts < 5; attempts++)
    {
        z_stream strm = {0};
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
        currsize = currthickness = -1.;
        scalefactor = 1.0;
        currunits = -1;
        ftface = 0;
    }

    ~Impl()
    {
        deleteFont();
    }

    void deleteFont()
    {
        if(ftface != 0)
            FT_Done_Face(ftface);
        ftface = 0;
        currname.clear();
    }

    void initParams()
    {
        currthickness = -1;
        currsize = -1;
        currunits = -1;
    }

    bool setStd(const String& name, double sf)
    {
        if(ftface != 0 && currname == name)
        {
            scalefactor = sf;
            return true;
        }
        FT_Library library = ftlib.getLibrary();
        if(library == 0) return false;
        deleteFont();

        int i = 0;
        for(; defaultFontData[i].name != 0; i++)
        {
            if(defaultFontData[i].name == name)
            {
                if(!inflate(defaultFontData[i].gzdata, defaultFontData[i].size, fontbuf))
                    return false;
                int err = FT_New_Memory_Face(library, &fontbuf[0], (FT_Long)fontbuf.size(), 0, &ftface);
                if(err != 0)
                    return false;
                break;
            }
        }
        if(defaultFontData[i].name == 0)
            return false;
        currname = name;
        scalefactor = sf;
        initParams();
        return true;
    }

    bool set(const String& fontname, double sf)
    {
        CV_Assert(!fontname.empty());

        if(ftface != 0 && fontname == currname)
        {
            scalefactor = sf;
            return true;
        }

        FT_Library library = ftlib.getLibrary();
        if(library == 0) return false;

        deleteFont();

        int err = FT_New_Face(library, fontname.c_str(), 0, &ftface);
        if(err != 0) { ftface = 0; return false; }
        currname = fontname;
        scalefactor = sf;
        initParams();
        return true;
    }

    bool setParams(double size, int thickness, int flags)
    {
        int sizeunits = flags & PUT_TEXT_SIZE_MASK;
        if (ftface == 0)
            return false;
        if (std::abs(size - currsize) < 1e-3 &&
            sizeunits == currunits &&
            thickness == currthickness)
            return true;

        FT_Library library = ftlib.getLibrary();
        if(library == 0) return false;

        FT_MM_Var* multimaster = 0;

        // retrieve the variable font information, if any
        if( thickness != currthickness &&
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
                    FT_Fixed th = cvRound((thickness <= 0 ? 400 : thickness)*(1<<16));
                    th = th < minval ? minval : th > maxval ? maxval : th;
                    design_pos[n] = th;
                }
            }
            FT_Set_Var_Design_Coordinates(ftface, multimaster->num_axis, design_pos);
        }

        if(multimaster)
            FT_Done_MM_Var(library, multimaster);
        if(size != currsize || sizeunits != currunits)
        {
            double actual_size = size*scalefactor;
            if(sizeunits == PUT_TEXT_SIZE_POINTS)
            {
                FT_Fixed sz = cvRound(actual_size*64);
                FT_Set_Char_Size(ftface, sz, sz, 120, 120);
            }
            else
            {
                FT_Set_Pixel_Sizes(ftface, 0, cvRound(actual_size));
            }
        }

        currthickness = thickness;
        currsize = size;
        currunits = sizeunits;
        return true;
    }

    String currname;
    double currsize;
    double scalefactor;
    int currunits;
    int currthickness;
    FT_Face ftface;
    std::vector<uchar> fontbuf;
};

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
    for( ; i < DEFAULT_FONTS_NUM; i++ )
    {
        if( defaultFontData[i].name == fontname )
            break;
    }
    if( i >= DEFAULT_FONTS_NUM )
        i = -1;

    bool ok;
    if( i >= 0 )
    {
        FontFace& def_fface = default_ffaces[i];
        ok = def_fface.impl->setStd(fontname, defaultFontData[i].sf);
        if(ok)
            impl = def_fface.impl;
    }
    else
    {
        sf = sf > 0 ? sf : 1.0;
        if(impl->ftface != 0)
            impl = makePtr<Impl>();
        ok = impl->set(fontname, sf);
    }
    return ok;
}

String FontFace::getName() const { return impl->currname; }
double FontFace::getScaleFactor() const { return impl->scalefactor; }
FontFace::Impl* FontFace::operator -> () { return impl.get(); }
FontFace::~FontFace() {}

static void drawCharacter(
    Mat& img, const uchar* color,
    const FT_Bitmap* bitmap, int x0, int y0 )
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
        int y = y0 + dy;
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
static bool isRightToLeft(int c)
{
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

static Point putText_( Mat& img, const String& str, Point org,
                       const uchar* color, FontFace& fontface, double size,
                       int thickness, int flags, bool render,
                       Rect* brect )
{
    int load_glyph_flag = render ? FT_LOAD_RENDER : FT_LOAD_DEFAULT;
    bool subst_initialized = false;

    if(fontface.getName().empty())
        fontface.set("sans");

    fontface->setParams(size, thickness, flags);
    FT_Face ftface = fontface->ftface;
    if(ftface == 0)
        return org;

    int pen_x = org.x, pen_y = org.y;
    size_t i, len = str.size();
    bool wrap = (flags & PUT_TEXT_WRAP) != 0;

    // text size computing algorithm is adopted from G-API module, ft_render.cpp.
    int max_dy = 0, max_baseline = 0;
    int max_width = 0;
    bool wrapped = false;

    int use_kerning = 0;//FT_HAS_KERNING(ftface);
    int prev_glyph = 0;
    bool rtl_mode = false;
    std::vector<int> rtl;

    for( i = 0; i < len; i++ )
    {
        uchar ch = (uchar)str[i];
        int charcode = 0;
        /*if( ch == 'f' )
        {
            uchar nch = (uchar)str[i+1];
            if(nch == 'i')
            {
                charcode = 0xFB01;
                i++;
            }
            else if(nch == 'l')
            {
                charcode = 0xFB02;
                i++;
            }
            else
                charcode = 'f';
        }
        else*/ if( ch <= 127 )
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
        else {
            charcode = 65533;
            while(i+1 < len && (str[i+1] & 0xc0) == 0x80)
                i++;
        }
        bool is_rtl = charcode >= 0x05BE && isRightToLeft(charcode);
        if(rtl_mode && (isspace(charcode) || ispunct(charcode)))
            is_rtl = true;
        const int* charcodes = &charcode;
        int nchars = 1;

        if(is_rtl || rtl_mode)
        {
            if(is_rtl)
            {
                if(!rtl_mode)
                {
                    rtl_mode = true;
                    rtl.clear();
                }
                rtl.push_back(charcode);
                if( i+1 < len )
                    continue;
            }
            rtl_mode = false;
            std::reverse(rtl.begin(), rtl.end());
            if(!is_rtl)
                rtl.push_back(charcode);
            nchars = (int)rtl.size();
            charcodes = nchars > 0 ? &rtl[0] : 0;
        }

        for(int k = 0; k < nchars; k++)
        {
            charcode = charcodes[k];
            FT_Face curr_ftface = ftface;
            FT_UInt glyph_index = FT_Get_Char_Index( curr_ftface, charcode );
            if (use_kerning && glyph_index > 0 && prev_glyph > 0)
            {
                FT_Vector delta;
                FT_Get_Kerning( curr_ftface, prev_glyph, glyph_index, FT_KERNING_DEFAULT, &delta );
                pen_x += delta.x >> 6;
            }
            prev_glyph = glyph_index;
            if(glyph_index == 0)
            {
                if(!subst_initialized)
                {
                    for(int j = 0; j < DEFAULT_FONTS_NUM; j++)
                    {
                        default_ffaces[j].set(defaultFontData[j].name);
                        default_ffaces[j]->setParams(size, thickness, flags);
                    }
                    subst_initialized = true;
                }
                for(int j = 0; j < DEFAULT_FONTS_NUM; j++)
                {
                    curr_ftface = default_ffaces[j]->ftface;
                    glyph_index = FT_Get_Char_Index( curr_ftface, charcode );
                    if(glyph_index != 0)
                        break;
                    if(j+1 == DEFAULT_FONTS_NUM)
                        glyph_index = FT_Get_Char_Index( curr_ftface, 0xFFFD );
                }
            }
            int err = FT_Load_Glyph( curr_ftface, glyph_index, load_glyph_flag );
            if( err != 0 )
                continue;

            FT_GlyphSlot slot = curr_ftface->glyph;

            int dx_shift = 6, dx_scale = 1;
            /*if(charcode == ' ')
            {
                dx_scale = 3;
                dx_shift++;
            }*/
            int dx_delta = 1 << (dx_shift - 1);
            int dx = (int)((slot->metrics.horiAdvance*dx_scale + dx_delta) >> dx_shift);
            int new_pen_x = pen_x + dx;
            if( wrap && img.cols > 0 && (new_pen_x > img.cols) )
            {
                pen_y += (int)((slot->metrics.vertAdvance + 32) >> 6);
                max_width = max(max_width, pen_x - org.x);
                pen_x = org.x;
                wrapped = true;
                max_baseline = slot->bitmap_top;
                if(charcode == ' ') continue;
                new_pen_x = pen_x + dx;
            }

            if(!wrapped)
                max_dy = std::max(max_dy, slot->bitmap_top);
            int baseline = (slot->metrics.height - slot->metrics.horiBearingY) >> 6;
            max_baseline = std::max(max_baseline, baseline);

            int x = pen_x + slot->bitmap_left;
            int y = pen_y - slot->bitmap_top;
            if( render )
                drawCharacter( img, color, &slot->bitmap, x, y );
            pen_x = new_pen_x;
        }
    }
    max_width = max(max_width, pen_x - org.x);

    if(brect)
        *brect = Rect(org.x, org.y - max_dy, max_width, pen_y - org.y + max_dy + max_baseline);

    return Point(pen_x, pen_y);
}


Point putText(InputOutputArray img_, const String& str,
             Point org, Scalar color_,
             FontFace& fontface, double size,
             int thickness, int flags)
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

    return putText_(img, str, org, color, fontface, size,
                    thickness, flags, true, 0);
}


Rect getTextSize(InputArray img_, const String& str, Point org,
                 FontFace& fontface, double size,
                 int thickness, int flags)
{
    Mat img = img_.getMat();
    Rect brect;
    putText_(img, str, org, 0, fontface, size,
             thickness, flags, false, &brect);
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

FontFace::~FontFace() {}
FontFace::Impl* FontFace::operator -> () { return impl.get(); }

Point putText(InputOutputArray, const String&, Point org, Scalar,
             FontFace&, double, int, int)
{
    CV_Error(Error::StsNotImplemented, "putText needs freetype2; recompile OpenCV with freetype2 enabled.");
    return org;
}

Rect getTextSize(InputArray, const String&, Point, FontFace&, double, int, int)
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
        sf = 3.3;
        ttweight = thickness <= 1 ? 400 : 800;
        break;
    case FONT_HERSHEY_TRIPLEX:
        ttname = "serif";
        sf = 3.3;
        ttweight = thickness <= 1 ? 600 : 800;
        break;
    case FONT_HERSHEY_COMPLEX_SMALL:
        ttname = "serif";
        sf = 4.6;
        ttweight = thickness <= 1 ? 400 : 800;
        break;
    case FONT_HERSHEY_SCRIPT_COMPLEX:
        ttname = "italic";
        sf = 4.2;
        ttweight = thickness <= 1 ? 400 : 800;
        break;
    case FONT_HERSHEY_SCRIPT_SIMPLEX:
        ttname = "italic";
        sf = 4.2;
        ttweight = thickness <= 1 ? 300 : 600;
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
    Rect r = getTextSize(noArray(), text, Point(), fface, ttsize, ttweight, 0);

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
