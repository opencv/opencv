class JavaStreamReader : public cv::IStreamReader
{
public:
    JavaStreamReader(jclass _jobject) : jobject(_jobject) {}

    long long read(char* buffer, long long size) CV_OVERRIDE
    {
        printf("native read\n");
        return 0;
    }

    long long seek(long long offset, int way) CV_OVERRIDE
    {
        printf("native seek\n");
        return 0;
    }

private:
    jclass jobject;
};
