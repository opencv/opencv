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
} DefaultFontData;

static DefaultFontData OcvDefaultFonts[] =
{
    {OcvDefaultFontUni, sizeof(OcvDefaultFontUni), "uni"},
    {OcvDefaultFontSans, sizeof(OcvDefaultFontSans), "sans"},
    {OcvDefaultFontSerif, sizeof(OcvDefaultFontSerif), "serif"},
    {OcvDefaultFontItalic, sizeof(OcvDefaultFontItalic), "italic"},
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
static thread_local FontFace def_font_uni, def_font_sans, def_font_serif, def_font_italic;

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

    bool setStd(const String& name)
    {
        if(ftface != 0 && currname == name)
            return true;
        FT_Library library = ftlib.getLibrary();
        if(library == 0) return false;
        deleteFont();

        int i = 0;
        for(; OcvDefaultFonts[i].name != 0; i++)
        {
            if(OcvDefaultFonts[i].name == name)
            {
                if(!inflate(OcvDefaultFonts[i].gzdata, OcvDefaultFonts[i].size, fontbuf))
                    return false;
                int err = FT_New_Memory_Face(library, &fontbuf[0], (FT_Long)fontbuf.size(), 0, &ftface);
                if(err != 0)
                    return false;
                break;
            }
        }
        if(OcvDefaultFonts[i].name == 0)
            return false;
        currname = name;
        initParams();
        return true;
    }

    bool set(const String& fontname)
    {
        CV_Assert(!fontname.empty());

        if(ftface != 0 && fontname == currname)
            return true;

        FT_Library library = ftlib.getLibrary();
        if(library == 0) return false;

        deleteFont();

        int err = FT_New_Face(library, fontname.c_str(), 0, &ftface);
        if(err != 0) { ftface = 0; return false; }
        currname = fontname;
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
            if(sizeunits == PUT_TEXT_SIZE_POINTS)
            {
                FT_Fixed sz = cvRound(size*64);
                FT_Set_Char_Size(ftface, sz, sz, 120, 120);
            }
            else
            {
                FT_Set_Pixel_Sizes(ftface, 0, cvRound(size));
            }
        }

        currthickness = thickness;
        currsize = size;
        currunits = sizeunits;
        return true;
    }

    String currname;
    double currsize;
    int currunits;
    int currthickness;
    FT_Face ftface;
    std::vector<uchar> fontbuf;
};

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
    if(impl->ftface != 0 && impl->currname == fontname)
        return true;
    bool ok;
    FontFace* def_fface = fontname == "uni" ? &def_font_uni :
                          fontname == "sans" ? &def_font_sans :
                          fontname == "serif" ? &def_font_serif :
                          fontname == "italic" ? &def_font_italic : 0;
    if( def_fface != 0 )
    {
        ok = def_fface->impl->setStd(fontname);
        if(ok)
            impl = def_fface->impl;
    }
    else
    {
        if(impl->ftface != 0)
            impl = makePtr<Impl>();
        ok = impl->set(fontname);
    }
    return ok;
}

String FontFace::get() const
{
    return impl->currname;
}

FontFace::Impl* FontFace::operator -> ()
{
    return impl.get();
}

FontFace::~FontFace()
{
}

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


static Point putText_( Mat& img, const String& str, Point org,
                       const uchar* color, FontFace& fontface, double size,
                       int thickness, int flags, bool render,
                       Rect* brect )
{
    int load_glyph_flag = render ? FT_LOAD_RENDER : FT_LOAD_DEFAULT;

    FontFace& subst = def_font_uni;
    if(subst.get().empty())
        subst.set("uni");

    if(fontface.get().empty())
        fontface.set("sans");

    subst->setParams(size, thickness, flags);
    fontface->setParams(size, thickness, flags);
    FT_Face subst_ftface = subst->ftface;
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

    for( i = 0; i < len; i++ )
    {
        uchar ch = (uchar)str[i];
        int charcode = 0;
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
        else {
            charcode = 65533;
            while(i+1 < len && (str[i+1] & 0xc0) == 0x80)
                i++;
        }

        FT_Face curr_ftface = ftface;
        FT_UInt glyph_index = FT_Get_Char_Index( curr_ftface, charcode );
        if(glyph_index == 0 && subst_ftface != 0 && subst_ftface != ftface)
        {
            curr_ftface = subst_ftface;
            glyph_index = FT_Get_Char_Index( curr_ftface, charcode );
            if(glyph_index == 0)
                glyph_index = FT_Get_Char_Index( curr_ftface, 0xFFFD );
        }
        int err = FT_Load_Glyph( curr_ftface, glyph_index, load_glyph_flag );
        if( err != 0 )
            continue;

        FT_GlyphSlot slot = curr_ftface->glyph;

        int dx_shift = 6, dx_scale = 1;
        if(charcode == ' ')
        {
            dx_scale = 3;
            dx_shift++;
        }
        int dx_delta = (1 << dx_shift) - 1;
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
    max_width = max(max_width, pen_x - org.x);

    printf("pen_y - org.y = %d\n", pen_y - org.y);
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
FontFace::FontFace(const String&) {}

bool FontFace::set(const String&) { return false; }
String FontFace::get() const { return String(); }

FontFace::~FontFace() {}
FontFace::Impl* FontFace::operator -> () { return impl.get(); }

Point putText(InputOutputArray, const String&, Point org, Scalar,
             FontFace&, double, int, int)
{
    CV_Error(Error::StsNotImplemented, "putText needs freetype2; recompile OpenCV with freetype2 enabled");
    return org;
}

Rect getTextSize(InputArray, const String&, Point, FontFace&, double, int, int)
{
    CV_Error(Error::StsNotImplemented, "putText needs freetype2; recompile OpenCV with freetype2 enabled");
    return Rect();
}

}
#endif
