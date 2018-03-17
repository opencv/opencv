#include "opencv2/core.hpp"
#include <iostream>
#include <string>

static CvFileStorage * three_same_ways_of_write_base64()
{
    CvFileStorage * fs = 0;
    cv::RNG rng;
    switch ( rng.uniform( 0, 2 ) )
    {
    case 0:
        //! [suffix_in_file_name]
        fs = cvOpenFileStorage( "example.yml?base64", 0, CV_STORAGE_WRITE );
        //! [suffix_in_file_name]
        break;
    case 1:
        //! [flag_write_base64]
        fs = cvOpenFileStorage( "example.yml"       , 0, CV_STORAGE_WRITE_BASE64 );
        //! [flag_write_base64]
        break;
    case 2:
        //! [flag_write_and_flag_base64]
        fs = cvOpenFileStorage( "example.yml"       , 0, CV_STORAGE_WRITE | CV_STORAGE_BASE64 );
        //! [flag_write_and_flag_base64]
        break;
    default:
        break;
    }
    return fs;
}

static void two_ways_to_write_rawdata_in_base64()
{
    std::vector<int> rawdata(10, 0x00010203);

    {   // [1]
        //! [without_base64_flag]
        CvFileStorage* fs = cvOpenFileStorage( "example.xml", 0, CV_STORAGE_WRITE );
        // both CV_NODE_SEQ and "binary" are necessary.
        cvStartWriteStruct(fs, "rawdata", CV_NODE_SEQ | CV_NODE_FLOW, "binary");
        cvWriteRawDataBase64(fs, rawdata.data(), static_cast<int>(rawdata.size()), "i");
        cvEndWriteStruct(fs);
        cvReleaseFileStorage( &fs );
        //! [without_base64_flag]
    }

    {   // [2]
        //! [with_write_base64_flag]
        CvFileStorage* fs = cvOpenFileStorage( "example.xml", 0, CV_STORAGE_WRITE_BASE64);
        // parameter, typename "binary" could be omitted.
        cvStartWriteStruct(fs, "rawdata", CV_NODE_SEQ | CV_NODE_FLOW);
        cvWriteRawData(fs, rawdata.data(), static_cast<int>(rawdata.size()), "i");
        cvEndWriteStruct(fs);
        cvReleaseFileStorage( &fs );
        //! [with_write_base64_flag]
    }
}

int main(int /* argc */, char** /* argv */)
{
    {   // base64 mode
        CvFileStorage * fs = three_same_ways_of_write_base64();
        cvReleaseFileStorage( &fs );
    }

    {   // output rawdata by `cvWriteRawdata*`
        two_ways_to_write_rawdata_in_base64();
    }

    return 0;
}
