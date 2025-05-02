class JavaStreamReader : public cv::IStreamReader
{
    long long read(char* buffer, long long size) CV_OVERRIDE
    {
        return 0;
    }

    long long seek(long long offset, int way) CV_OVERRIDE
    {
        return 0;
    }
};
