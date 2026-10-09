#ifndef OPENCV_ARMPL_HAL_RUNTIME_HPP
#define OPENCV_ARMPL_HAL_RUNTIME_HPP

void *armpl_hal_get_function(const char *name);

class ArmplSingleThread
{
public:
    explicit ArmplSingleThread(double work);
    ~ArmplSingleThread();
private:
    int nthreads;
};

#endif
