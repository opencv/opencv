// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#ifdef HAVE_FREETYPE

#include "ft2build.h"
#include FT_FREETYPE_H
#include FT_MULTIPLE_MASTERS_H

namespace cv
{

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

class Font::Impl {
public:
    Impl()
    {
        font = 0;
        currsize = currthickness = -1.;
        currunits = SizePt;
        curritalic = false;
    }

    ~Impl()
    {
        if(font)
        {
            FT_Done_Face(font);
            font = 0;
        }
    }

    bool set(const String& fontface,
             double size, int sizeunits,
             double thickness, bool italic)
    {
        int err = 0;

        if(font != 0 && fontface == currface &&
           std::abs(size - currsize) < 1e-3 &&
           sizeunits == currunits &&
           thickness == currthickness &&
           italic == curritalic)
            return true;

        FT_Library library = ftlib.getLibrary();
        if(library == 0) return false;
        if (font == 0 || fontface != currface)
        {
            if(font) FT_Done_Face(font);
            err = FT_New_Face(library, fontface.c_str(), 0, &font);
            if(err != 0) { font = 0; return false; }
            currface = fontface;
            currthickness = -1;
            curritalic = false;
            currsize = -1;
            currunits = SizePt;
        }

        FT_MM_Var* multimaster = 0;

        // retrieve the variable font information, if any
        if( (thickness != currthickness || italic != curritalic) &&
            FT_Get_MM_Var( font, &multimaster ) == 0 )
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
                    FT_Fixed th = cvRound(thickness*(1<<16));
                    th = th < minval ? minval : th > maxval ? maxval : th;
                    design_pos[n] = th;
                }
                else if(strcmp(name, "Slant") == 0)
                {
                    if(italic)
                        design_pos[n] = minval;
                    else
                        design_pos[n] = 0;
                }
            }
            FT_Set_Var_Design_Coordinates( font, multimaster->num_axis, design_pos);
        }

        if(multimaster)
            FT_Done_MM_Var(library, multimaster);

        currthickness = thickness;
        curritalic = italic;

        if(size != currsize || sizeunits != currunits)
        {
            if(sizeunits == SizePt)
            {
                FT_Fixed sz = cvRound(size*64);
                FT_Set_Char_Size(font, sz, sz, 120, 120);
            }
            else
            {
                FT_Set_Pixel_Sizes(font, 0, cvRound(size));
            }
        }
        currsize = size;
        currunits = sizeunits;
        return true;
    }

    String currface;
    double currsize;
    int currunits;
    int currthickness;
    bool curritalic;
    FT_Face font;
};

Font::Font() { impl = makePtr<Impl>(); }
Font::Font(const String& fontface,
           double size, int sizeUnits,
           double thickness, bool italic)
{
    impl = makePtr<Impl>();
    impl->set(fontface, size, sizeUnits, thickness, italic);
}

bool Font::set(const String& fontface,
               double size, int sizeUnits,
               double thickness, bool italic)
{
    return impl->set(fontface, size, sizeUnits, thickness, italic);
}

Font::~Font()
{
}

void* Font::handle() const { return impl->font; }

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


void putText(InputOutputArray img_, const String& str, Point org,
             const Font& font, Scalar color_, char, bool)
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
    FT_Face ftface = (FT_Face)font.handle();
    if(ftface == 0)
        return;

    int pen_x = org.x, pen_y = org.y;
    size_t i, len = str.size();

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

        FT_UInt glyph_index = FT_Get_Char_Index( ftface, charcode );
        if(glyph_index == 0)
            glyph_index = FT_Get_Char_Index( ftface, '?' );
        int err = FT_Load_Glyph( ftface, glyph_index, FT_LOAD_RENDER );
        if( err != 0 )
            continue;

        FT_GlyphSlot slot = ftface->glyph;

        int dx = (slot->metrics.horiAdvance >> 6);
        int new_pen_x = pen_x + dx;
        if( new_pen_x > img.cols - 50 )
        {
            pen_y += (int)((slot->metrics.vertAdvance >> 6)*1.0);
            pen_x = org.x;
            if(charcode == ' ') continue;
            new_pen_x = pen_x + dx;
        }

        int x = pen_x + slot->bitmap_left;
        int y = pen_y - slot->bitmap_top;
        drawCharacter( img, color, &slot->bitmap, x, y );
        pen_x = new_pen_x;
    }
}

Size getTextSize(const String& str, const Font& font, int* baseline)
{
    return Size();
}

}
#else
namespace cv
{
    
class Impl {
public:
    Impl() {}
};

Font::Font() {}
Font::Font(const String&,
           double, int,
           double, bool)
{}

bool Font::set(const String&, double, int,
               double, bool)
{ return false; }
    
Font::~Font() {}
void* Font::handle() const { return 0; }

void putText(InputOutputArray, const String&, Point,
             const Font&, Scalar, char, bool)
{
    CV_Error(Error::StsNotImplemented, "putText needs freetype2; recompile OpenCV with freetype2 enabled");
}

Size getTextSize(const String&, const Font&, int*)
{
    CV_Error(Error::StsNotImplemented, "putText needs freetype2; recompile OpenCV with freetype2 enabled");
    return Size();
}

}
#endif
