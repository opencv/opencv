// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

#include "opencv2/core/utils/filesystem.hpp"

// The drawing tests (and their reference pictures) were made with the "italic"
// (Rubik Italic) and "uni" (WenQuanYi Micro Hei) fonts compiled into OpenCV. Both
// are optional now (WITH_ITALICFONT / WITH_UNIFONT, OFF by default), so provide
// them as external files from opencv_extra when they are available.
static void registerExternalTestFonts()
{
    const std::string dir = cvtest::TS::ptr()->get_data_path() + "../highgui/drawing/";
    const char* fonts[][2] = {{"italic", "Rubik-Italic.ttf.gz"}, {"uni", "WenQuanYiMicroHei.ttf.gz"}};
    for (const auto& f : fonts)
    {
        const std::string path = dir + f[1];
        if (cv::utils::fs::exists(path))
            cv::FontFace::setBuiltinFont(f[0], path);
    }
}

CV_TEST_MAIN("cv", registerExternalTestFonts())
