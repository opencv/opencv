#ifndef __SFT_RANDOM_HPP__
#define __SFT_RANDOM_HPP__

#if defined(_MSC_VER) && _MSC_VER >= 1600

# include <random>
namespace sft {
struct Random
{
    typedef std::mt19937 engine;
    typedef std::uniform_int<int> uniform;
};
}

#elif (__GNUC__) && __GNUC__ > 3 && __GNUC_MINOR__ > 1

# if defined (__cplusplus) && __cplusplus > 201100L
#  include <random>
namespace sft {
struct Random
{
    typedef std::mt19937 engine;
    typedef std::uniform_int<int> uniform;
};
}
# else
#   include <tr1/random>

namespace sft {
struct Random
{
    typedef std::tr1::mt19937 engine;
    typedef std::tr1::uniform_int<int> uniform;
};
}
# endif

#else
#include <opencv2/core/core.hpp>
namespace rnd {

typedef cv::RNG engine;

template<typename T>
struct uniform_int
{
    uniform_int(const int _min, const int _max) : min(_min), max(_max) {}
    T operator() (engine& eng, const int bound) const
    {
        return (T)eng.uniform(min, bound);
    }

    T operator() (engine& eng) const
    {
        return (T)eng.uniform(min, max);
    }

private:
    int min;
    int max;
};

}

namespace sft {
struct Random
{
    typedef rnd::engine engine;
    typedef rnd::uniform_int<int> uniform;
};
}

#endif

#endif