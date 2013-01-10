/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <sft/fpool.hpp>
#include <sft/random.hpp>

#include <queue>

// ========= FeaturePool ========= //

sft::ICFFeaturePool::ICFFeaturePool(cv::Size m, int n) : FeaturePool(), model(m), nfeatures(n)
{
    CV_Assert(m != cv::Size() && n > 0);
    fill(nfeatures);
}

void sft::ICFFeaturePool::preprocess(cv::InputArray frame, cv::OutputArray integrals) const
{
    preprocessor.apply(frame, integrals);
}

float sft::ICFFeaturePool::apply(int fi, int si, const Mat& integrals) const
{
    return pool[fi](integrals.row(si), model);
}

void sft::ICFFeaturePool::write( cv::FileStorage& fs, int index) const
{
    CV_Assert((index > 0) && (index < (int)pool.size()));
    fs << pool[index];
}

void sft::write(cv::FileStorage& fs, const string&, const ICF& f)
{
    fs << "{" << "channel" << f.channel << "rect" << f.bb << "}";
}

sft::ICFFeaturePool::~ICFFeaturePool(){}

// #define USE_LONG_SEEDS
#if defined USE_LONG_SEEDS
# define FEATURE_RECT_SEED      8854342234LU
#else
# define FEATURE_RECT_SEED      88543422LU
#endif
# define DCHANNELS_SEED         314152314LU
#undef USE_LONG_SEEDS

void sft::ICFFeaturePool::fill(int desired)
{
    int mw = model.width;
    int mh = model.height;

    int maxPoolSize = (mw -1) * mw / 2 * (mh - 1) * mh / 2 * N_CHANNELS;

    nfeatures = std::min(desired, maxPoolSize);
    dprintf("Requeste feature pool %d max %d suggested %d\n", desired, maxPoolSize, nfeatures);

    pool.reserve(nfeatures);

    sft::Random::engine eng(FEATURE_RECT_SEED);
    sft::Random::engine eng_ch(DCHANNELS_SEED);

    sft::Random::uniform chRand(0, N_CHANNELS - 1);

    sft::Random::uniform xRand(0, model.width  - 2);
    sft::Random::uniform yRand(0, model.height - 2);

    sft::Random::uniform wRand(1, model.width  - 1);
    sft::Random::uniform hRand(1, model.height - 1);

    while (pool.size() < size_t(nfeatures))
    {
        int x = xRand(eng);
        int y = yRand(eng);

        int w = 1 + wRand(eng, model.width  - x - 1);
        int h = 1 + hRand(eng, model.height - y - 1);

        CV_Assert(w > 0);
        CV_Assert(h > 0);

        CV_Assert(w + x < model.width);
        CV_Assert(h + y < model.height);

        int ch = chRand(eng_ch);

        sft::ICF f(x, y, w, h, ch);

        if (std::find(pool.begin(), pool.end(),f) == pool.end())
        {
            pool.push_back(f);
            std::cout << f << std::endl;
        }
    }
}

std::ostream& sft::operator<<(std::ostream& out, const sft::ICF& m)
{
    out << m.channel << " " << m.bb;
    return out;
}

// ============ Dataset ============ //
namespace {
using namespace sft;

string itoa(long i)
{
    char s[65];
    sprintf(s, "%ld", i);
    return std::string(s);
}

}

#if !defined (_WIN32) && ! defined(__MINGW32__)

#include <glob.h>

namespace {
using namespace sft;
void glob(const string& path, svector& ret)
{
    glob_t glob_result;
    glob(path.c_str(), GLOB_TILDE, 0, &glob_result);

    ret.clear();
    ret.reserve(glob_result.gl_pathc);

    for(unsigned int i = 0; i < glob_result.gl_pathc; ++i)
    {
        ret.push_back(std::string(glob_result.gl_pathv[i]));
        dprintf("%s\n", ret[i].c_str());
    }

    globfree(&glob_result);
}

}
#else

#include <windows.h>
namespace {
using namespace sft;
void glob(const string& refRoot, const string& refExt, svector &refvecFiles)
{
    std::string     strFilePath;             // Filepath
    std::string     strExtension;            // Extension

    std::string strPattern = refRoot + "\\*.*";

    WIN32_FIND_DATA FileInformation;         // File information
    HANDLE hFile = ::FindFirstFile(strPattern.c_str(), &FileInformation);

    if(hFile == INVALID_HANDLE_VALUE)
        CV_Error(CV_StsBadArg, "Your dataset search path is incorrect");

    do
    {
        if(FileInformation.cFileName[0] != '.')
        {
            strFilePath.erase();
            strFilePath = refRoot + "\\" + FileInformation.cFileName;

            if( !(FileInformation.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) )
            {
                // Check extension
                strExtension = FileInformation.cFileName;
                strExtension = strExtension.substr(strExtension.rfind(".") + 1);

                if(strExtension == refExt)
                    // Save filename
                    refvecFiles.push_back(strFilePath);
            }
        }
    }
    while(::FindNextFile(hFile, &FileInformation) == TRUE);

    // Close handle
    ::FindClose(hFile);

    DWORD dwError = ::GetLastError();
    if(dwError != ERROR_NO_MORE_FILES)
        CV_Error(CV_StsBadArg, "Your dataset search path is incorrect");
}
}

#endif

// in the default case data folders should be alligned as following:
// 1. positives: <train or test path>/octave_<octave number>/pos/*.png
// 2. negatives: <train or test path>/octave_<octave number>/neg/*.png
ScaledDataset::ScaledDataset(const string& path, const int oct)
{
    dprintf("%s\n", "get dataset file names...");

    dprintf("%s\n", "Positives globbing...");

#if !defined (_WIN32) && ! defined(__MINGW32__)
    glob(path + "/pos/octave_" + itoa(oct) + "/*.png", pos);
#else
    glob(path + "/pos/octave_" + itoa(oct),     "png", pos);
#endif

    dprintf("%s\n", "Negatives globbing...");
#if !defined (_WIN32) && ! defined(__MINGW32__)
    glob(path + "/neg/octave_" + itoa(oct) + "/*.png", neg);
#else
    glob(path + "/neg/octave_" + itoa(oct),     "png", neg);
#endif

    // Check: files not empty
    CV_Assert(pos.size() != size_t(0));
    CV_Assert(neg.size() != size_t(0));
}

cv::Mat ScaledDataset::get(SampleType type, int idx) const
{
    const std::string& src = (type == POSITIVE)? pos[idx]: neg[idx];
    return cv::imread(src);
}

int ScaledDataset::available(SampleType type) const
{
    return (int)((type == POSITIVE)? pos.size():neg.size());
}

ScaledDataset::~ScaledDataset(){}