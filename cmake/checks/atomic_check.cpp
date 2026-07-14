#include <atomic>

static int test()
{
    std::atomic<long long> x;
    return x;
}

int main()
{
    return test();
}
