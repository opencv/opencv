// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace cv {

/** Class for drawing OpenCV logo

OpenCV logo consists of three "C" letters at different angles placed in corners of a triangle.
*/
class Logo
{
public:
    //! Location for drawing the logo
    enum Location
    {
        BottomRight,
        BottomLeft,
        TopRight,
        TopLeft
    };
    //! Color scheme for the logo
    enum Scheme
    {
        Classic,
        Modern
    };

protected:
    int d1; // inner "C" diameter
    int d2; // outer "C" diameter
    int l; // side of the triangle - distance between "C"'s centers
    std::vector<Point> cb, cg, cr; // contours for each "C"
    Scalar b, g, r; // colors for drawing each "C"

protected:
    inline std::vector<Point> getPoly(int angle) const
    {
        const int r1 = static_cast<int>(d1 / 2.f);
        const int r2 = static_cast<int>(d2 / 2.f);
        const int delta = std::max(1, static_cast<int>(360. / d2)); // 3.14 px/degree
        std::vector<Point> pts1;
        std::vector<Point> pts2;
        ellipse2Poly(Point(0, 0), Size(r1, r1), 0, angle, angle + 300, delta, pts1);
        ellipse2Poly(Point(0, 0), Size(r2, r2), 0, angle, angle + 300, delta, pts2);
        std::copy(pts1.rbegin(), pts1.rend(), std::back_inserter(pts2));
        return pts2;
    }
    inline Point getOrigin(const Size& imageSize, Location loc) const
    {
        const Point br = imageSize - size() - Size(2, 2);
        switch (loc)
        {
        case BottomRight: return br;
        case TopLeft: return Point(2, 2);
        case BottomLeft: return br + Point(2, 0);
        case TopRight: return br + Point(0, 2);
        }
        return Point(0, 0);
    }
    inline void init(float scale)
    {
        CV_Assert(scale > 0);
        d1 = static_cast<int>(10 * scale);
        d2 = static_cast<int>(25 * scale);
        l = static_cast<int>(28 * scale);
        CV_Assert(d1 > 0 && d2 > 0 && l > 0 && d2 > d1 && l >= d2);
        cb = getPoly(-60);
        cg = getPoly(0);
        cr = getPoly(-240);
        setColorScheme(Modern);
    }

public:
    //! Create logo object with specified scale
    //! @param scale logo ize multiplier, scale=1.0 means logo will be about 50x50 px
    Logo(float scale = 1.f)
    {
        init(scale);
    }
    //! Create logo object for specific image size
    //! @param imageSize size of an image
    //! @param factor logo size multiplier, factor=1.0 means logo wlil be slightly less then min image dimension (W or H)
    Logo(const Size& imageSize, float factor = 0.1f)
    {
        CV_Assert(factor > 0 && !imageSize.empty());
        const float scale = std::min(imageSize.width, imageSize.height) / 52.f * factor;
        init(scale);
    }
    //! Returns estimated size of the logo
    Size size() const
    {
        const float h = l * sqrt(3.f) / 2.f;
        return Size(d2 + l, static_cast<int>(d2 + h));
    }
    //! Changes color scheme of the logo
    //! @param scheme one of Logo::Scheme
    void setColorScheme(Scheme scheme)
    {
        switch (scheme)
        {
        case Modern:
            b = Scalar(0xFF, 0x8D, 0x12);
            g = Scalar(0x67, 0xDA, 0x8B);
            r = Scalar(0x44, 0x2A, 0xFF);
            break;
        case Classic:
            b = Scalar(0xff, 0x01, 0x01);
            g = Scalar(0x01, 0xff, 0x01);
            r = Scalar(0x01, 0x01, 0xff);
            break;
        }
    }
    //! Draw logo on an image with specified origin point
    //! @param image image for drawing
    //! @param origin origin point for drawing (top left corner)
    void draw(InputOutputArray image, const Point& origin) const
    {
        CV_Assert(!image.empty());
        const float r2 = d2 / 2.f;
        const float h = l * sqrt(3.f) / 2.f;
        fillPoly(image, cb, b, LINE_AA, 0, origin + Point(static_cast<int>(r2 + l), static_cast<int>(r2 + h)));
        fillPoly(image, cg, g, LINE_AA, 0, origin + Point(static_cast<int>(r2), static_cast<int>(r2 + h)));
        fillPoly(image, cr, r, LINE_AA, 0, origin + Point(static_cast<int>(r2 + l / 2.f), static_cast<int>(r2)));
    }
    //! Draw logo on an image with automatic placement
    //! @param image image for drawing
    //! @param loc location for drawing, one of Logo::Location
    void draw(InputOutputArray image, Logo::Location loc = BottomRight) const
    {
        draw(image, getOrigin(image.size(), loc));
    }
    //! Create logo object and draw it in the bottom right corner of an image
    //! @param image image for drawing
    //! @param factor logo size multiplier, factor=1.0 means logo wlil be slightly less then min image dimension (W or H)
    static void drawBR(InputOutputArray image, float factor = 0.15f)
    {
        Logo logo(image.size(), factor);
        logo.draw(image, BottomRight);
    }
};

} // cv::
