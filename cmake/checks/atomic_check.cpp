#include <atomic>

static int test()
{
    std::atomic<int> x;
    return x;
}

int main()
{
    return test();
}
