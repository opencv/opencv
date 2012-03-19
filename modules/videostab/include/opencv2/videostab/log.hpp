#ifndef __OPENCV_VIDEOSTAB_LOG_HPP__
#define __OPENCV_VIDEOSTAB_LOG_HPP__

namespace cv
{
namespace videostab
{

class ILog
{
public:
    virtual ~ILog() {}
    virtual void print(const char *format, ...) = 0;
};

class NullLog : public ILog
{
public:
    virtual void print(const char *format, ...) {}
};

class LogToStdout : public ILog
{
public:
    virtual void print(const char *format, ...);
};

} // namespace videostab
} // namespace cv

#endif
