namespace cv
{
enum RemapType {
    fp32_mapxy,
    fp32_mapx_mapy,
    fixedPointInt16,
    int16,
    errorType
};

static const std::string errorRemapMessage =
        "fp32_mapxy: map1 having the type CV_32FC2; map2 is empty\n"
        "fp32_mapx_mapy: map1 having the type CV_32FC1; map2 having the type CV_32FC1\n"
        "fixedPointInt16: map1 having the type CV_16SC2; map2 having the type CV_16UC1 or CV_16SC1\n"
        "int16: map1 having the type CV_16SC2; map2 is empty\n"
        "If map2 isn't empty, map1.size() must be equal to map2.size().\n";

inline RemapType get_remap_type(const Mat &map1, const Mat &map2)
{
    if ((map1.depth() == CV_32F && !map1.empty()) || (map2.depth() == CV_32F && !map2.empty())) // fp32_mapxy or fp32_mapx_mapy
    {
        if (map1.type() == CV_32FC2 && map2.empty())
            return RemapType::fp32_mapxy;
        else if (map1.type() == CV_32FC1 && map2.type() == CV_32FC1)
            return RemapType::fp32_mapx_mapy;
    }
    else if (map1.channels() == 2) // int or fixedPointInt
    {
        if (map2.empty()) // int16
        {
            if (map1.type() == CV_16SC2) // int16
                return RemapType::int16;
        }
        else if (map2.channels() == 1) // fixedPointInt16
        {
            if (map1.type() == CV_16SC2 && (map2.type() == CV_16UC1 || map2.type() == CV_16SC1)) // fixedPointInt16
                return RemapType::fixedPointInt16;
        }
    }
    return RemapType::errorType;
}

inline RemapType check_and_get_remap_type(Mat &map1, Mat &map2)
{
    CV_Assert(!map1.empty() || !map2.empty());
    if (map2.channels() == 2 && !map2.empty())
        std::swap(map1, map2);
    CV_Assert((map2.empty() && map1.channels() == 2) ||
              (!map2.empty() && map1.size() == map2.size() && map2.channels() == 1));

    RemapType type = get_remap_type(map1, map2);
    if (type == RemapType::errorType)
        CV_Error(cv::Error::StsBadSize, errorRemapMessage.data());
    return type;
}
}