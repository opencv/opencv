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
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
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
//     and / or other materials provided with the distribution.
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

#if !defined(ANDROID)

#include <string>
#include <fstream>
#include <vector>

#include "test_precomp.hpp"
#if !defined (_WIN32) && ! defined(__MINGW32__)
# include <glob.h>
#else
# include <windows.h>
#endif

using namespace std;

namespace {

using namespace cv::softcascade;

typedef vector<string> svector;
class ScaledDataset : public Dataset
{
public:
    ScaledDataset(const string& path, const int octave);

    virtual cv::Mat get(SampleType type, int idx) const;
    virtual int available(SampleType type) const;
    virtual ~ScaledDataset();

private:
    svector pos;
    svector neg;
};

string itoa(long i)
{
    char s[65];
    sprintf(s, "%ld", i);
    return std::string(s);
}


#if !defined (_WIN32) && ! defined(__MINGW32__)

void glob(const string& path, svector& ret)
{
    glob_t glob_result;
    glob(path.c_str(), GLOB_TILDE, 0, &glob_result);

    ret.clear();
    ret.reserve(glob_result.gl_pathc);

    for(unsigned int i = 0; i < glob_result.gl_pathc; ++i)
    {
        ret.push_back(std::string(glob_result.gl_pathv[i]));
    }

    globfree(&glob_result);
}

#else

void glob(const string& refRoot, const string& refExt, svector &refvecFiles)
{
    std::string     strFilePath;             // File path
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

#endif

ScaledDataset::ScaledDataset(const string& path, const int oct)
{

#if !defined (_WIN32) && ! defined(__MINGW32__)
    glob(path + "/pos/octave_" + itoa(oct) + "/*.png", pos);
#else
    glob(path + "/pos/octave_" + itoa(oct),     "png", pos);
#endif

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

}

TEST(DISABLED_SoftCascade, training)
{
    // // 2. check and open output file
    string outXmlPath = cv::tempfile(".xml");
    cv::FileStorage fso(outXmlPath, cv::FileStorage::WRITE);

    ASSERT_TRUE(fso.isOpened());

    std::vector<int> octaves;
    {
        octaves.push_back(-1);
        octaves.push_back(0);
    }

    fso << "regression-cascade"
        << "{"
        << "stageType"   << "BOOST"
        << "featureType" << "ICF"
        << "octavesNum"  << 2
        << "width"       << 64
        << "height"      << 128
        << "shrinkage"   << 4
        << "octaves"     << "[";

    for (std::vector<int>::const_iterator it = octaves.begin(); it != octaves.end(); ++it)
    {
        int nfeatures  = 100;
        int shrinkage = 4;
        float octave = powf(2.f, (float)(*it));
        cv::Size model = cv::Size( cvRound(64 * octave) / shrinkage, cvRound(128 * octave) / shrinkage );

        cv::Ptr<FeaturePool> pool = FeaturePool::create(model, nfeatures);
        nfeatures = pool->size();
        int npositives = 20;
        int nnegatives = 40;

        cv::Rect boundingBox = cv::Rect( cvRound(20 * octave), cvRound(20  * octave),
                                         cvRound(64 * octave), cvRound(128 * octave));

        cv::Ptr<Octave> boost = Octave::create(boundingBox, npositives, nnegatives, *it, shrinkage, nfeatures);

        std::string path = cvtest::TS::ptr()->get_data_path() + "softcascade/sample_training_set";
        ScaledDataset dataset(path, *it);

        if (boost->train(&dataset, pool, 3, 2))
        {
            cv::Mat thresholds;
            boost->setRejectThresholds(thresholds);
            boost->write(fso, pool, thresholds);
        }
    }

    fso << "]" << "}";
    fso.release();


    cv::FileStorage actual(outXmlPath, cv::FileStorage::READ);
    cv::FileNode root = actual.getFirstTopLevelNode();

    cv::FileNode fn = root["octaves"];
    ASSERT_FALSE(fn.empty());
}

#endif