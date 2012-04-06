#include "precomp.hpp"

using namespace perf;

int64 TestBase::timeLimitDefault = 0;
unsigned int TestBase::iterationsLimitDefault = (unsigned int)(-1);
int64 TestBase::_timeadjustment = 0;

const char *command_line_keys =
{
    "{   |perf_max_outliers   |8        |percent of allowed outliers}"
    "{   |perf_min_samples    |10       |minimal required numer of samples}"
    "{   |perf_force_samples  |100      |force set maximum number of samples for all tests}"
    "{   |perf_seed           |809564   |seed for random numbers generator}"
    "{   |perf_tbb_nthreads   |-1       |if TBB is enabled, the number of TBB threads}"
    "{   |perf_write_sanity   |false    |allow to create new records for sanity checks}"
    #if ANDROID
    "{   |perf_time_limit     |6.0      |default time limit for a single test (in seconds)}"
    "{   |perf_affinity_mask  |0        |set affinity mask for the main thread}"
    #else
    "{   |perf_time_limit     |3.0      |default time limit for a single test (in seconds)}"
    #endif
    "{   |perf_max_deviation  |1.0      |}"
    "{h  |help                |false    |}"
};

static double       param_max_outliers;
static double       param_max_deviation;
static unsigned int param_min_samples;
static unsigned int param_force_samples;
static uint64       param_seed;
static double       param_time_limit;
static int          param_tbb_nthreads;
static bool         param_write_sanity;
#if ANDROID
static int          param_affinity_mask;

#include <sys/syscall.h>
#include <pthread.h>
static void setCurrentThreadAffinityMask(int mask)
{
    pid_t pid=gettid();
    int syscallres=syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallres)
    {
        int err=errno;
        err=err;//to avoid warnings about unused variables
        LOGE("Error in the syscall setaffinity: mask=%d=0x%x err=%d=0x%x", mask, mask, err, err);
    }
}

#endif

void randu(cv::Mat& m)
{
    const int bigValue = 0x00000FFF;
    if (m.depth() < CV_32F)
    {
        int minmax[] = {0, 256};
        cv::Mat mr = cv::Mat(m.rows, (int)(m.cols * m.elemSize()), CV_8U, m.ptr(), m.step[0]);
        cv::randu(mr, cv::Mat(1, 1, CV_32S, minmax), cv::Mat(1, 1, CV_32S, minmax + 1));
    }
    else if (m.depth() == CV_32F)
    {
        //float minmax[] = {-FLT_MAX, FLT_MAX};
        float minmax[] = {-bigValue, bigValue};
        cv::Mat mr = m.reshape(1);
        cv::randu(mr, cv::Mat(1, 1, CV_32F, minmax), cv::Mat(1, 1, CV_32F, minmax + 1));
    }
    else
    {
        //double minmax[] = {-DBL_MAX, DBL_MAX};
        double minmax[] = {-bigValue, bigValue};
        cv::Mat mr = m.reshape(1);
        cv::randu(mr, cv::Mat(1, 1, CV_64F, minmax), cv::Mat(1, 1, CV_64F, minmax + 1));
    }
}

/*****************************************************************************************\
*                       inner exception class for early termination
\*****************************************************************************************/

class PerfEarlyExitException: public cv::Exception {};

/*****************************************************************************************\
*                                   ::perf::Regression
\*****************************************************************************************/

Regression& Regression::instance()
{
    static Regression single;
    return single;
}

Regression& Regression::add(const std::string& name, cv::InputArray array, double eps, ERROR_TYPE err)
{
    return instance()(name, array, eps, err);
}

void Regression::Init(const std::string& testSuitName, const std::string& ext)
{
    instance().init(testSuitName, ext);
}

void Regression::init(const std::string& testSuitName, const std::string& ext)
{
    if (!storageInPath.empty())
    {
        LOGE("Subsequent initialisation of Regression utility is not allowed.");
        return;
    }

    const char *data_path_dir = getenv("OPENCV_TEST_DATA_PATH");
    const char *path_separator = "/";

    if (data_path_dir)
    {
        int len = (int)strlen(data_path_dir)-1;
        if (len < 0) len = 0;
        std::string path_base = (data_path_dir[0] == 0 ? std::string(".") : std::string(data_path_dir))
                + (data_path_dir[len] == '/' || data_path_dir[len] == '\\' ? "" : path_separator)
                + "perf"
                + path_separator;

        storageInPath = path_base + testSuitName + ext;
        storageOutPath = path_base + testSuitName;
    }
    else
    {
        storageInPath = testSuitName + ext;
        storageOutPath = testSuitName;
    }

    try
    {
        if (storageIn.open(storageInPath, cv::FileStorage::READ))
        {
            rootIn = storageIn.root();
            if (storageInPath.length() > 3 && storageInPath.substr(storageInPath.length()-3) == ".gz")
                storageOutPath += "_new";
            storageOutPath += ext;
        }
    }
    catch(cv::Exception& ex)
    {
        LOGE("Failed to open sanity data for reading: %s", storageInPath.c_str());
    }
    
    if(!storageIn.isOpened())
        storageOutPath = storageInPath;
}

Regression::Regression() : regRNG(cv::getTickCount())//this rng should be really random
{
}

Regression::~Regression()
{
    if (storageIn.isOpened())
        storageIn.release();
    if (storageOut.isOpened())
    {
        if (!currentTestNodeName.empty())
            storageOut << "}";
        storageOut.release();
    }
}

cv::FileStorage& Regression::write()
{
    if (!storageOut.isOpened() && !storageOutPath.empty())
    {
        int mode = (storageIn.isOpened() && storageInPath == storageOutPath)
                ? cv::FileStorage::APPEND : cv::FileStorage::WRITE;
        storageOut.open(storageOutPath, mode);
        if (!storageOut.isOpened())
        {
            LOGE("Could not open \"%s\" file for writing", storageOutPath.c_str());
            storageOutPath.clear();
        }
        else if (mode == cv::FileStorage::WRITE && !rootIn.empty())
        {
            //TODO: write content of rootIn node into the storageOut
        }
    }
    return storageOut;
}

std::string Regression::getCurrentTestNodeName()
{
    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();

    if (test_info == 0)
        return "undefined";

    std::string nodename = std::string(test_info->test_case_name()) + "--" + test_info->name();
    size_t idx = nodename.find_first_of('/');
    if (idx != std::string::npos)
        nodename.erase(idx);

    const char* type_param = test_info->type_param();
    if (type_param != 0)
        (nodename += "--") += type_param;

    const char* value_param = test_info->value_param();
    if (value_param != 0)
        (nodename += "--") += value_param;

    for(size_t i = 0; i < nodename.length(); ++i)
        if (!isalnum(nodename[i]) && '_' != nodename[i])
            nodename[i] = '-';

    return nodename;
}

bool Regression::isVector(cv::InputArray a)
{
    return a.kind() == cv::_InputArray::STD_VECTOR_MAT || a.kind() == cv::_InputArray::STD_VECTOR_VECTOR;
}

double Regression::getElem(cv::Mat& m, int y, int x, int cn)
{
    switch (m.depth())
    {
    case CV_8U: return *(m.ptr<unsigned char>(y, x) + cn);
    case CV_8S: return *(m.ptr<signed char>(y, x) + cn);
    case CV_16U: return *(m.ptr<unsigned short>(y, x) + cn);
    case CV_16S: return *(m.ptr<signed short>(y, x) + cn);
    case CV_32S: return *(m.ptr<signed int>(y, x) + cn);
    case CV_32F: return *(m.ptr<float>(y, x) + cn);
    case CV_64F: return *(m.ptr<double>(y, x) + cn);
    default: return 0;
    }
}

void Regression::write(cv::Mat m)
{
    double min, max;
    cv::minMaxLoc(m, &min, &max);
    write() << "min" << min << "max" << max;

    write() << "last" << "{" << "x" << m.cols-1 << "y" << m.rows-1
        << "val" << getElem(m, m.rows-1, m.cols-1, m.channels()-1) << "}";

    int x, y, cn;
    x = regRNG.uniform(0, m.cols);
    y = regRNG.uniform(0, m.rows);
    cn = regRNG.uniform(0, m.channels());
    write() << "rng1" << "{" << "x" << x << "y" << y;
    if(cn > 0) write() << "cn" << cn;
    write() << "val" << getElem(m, y, x, cn) << "}";

    x = regRNG.uniform(0, m.cols);
    y = regRNG.uniform(0, m.rows);
    cn = regRNG.uniform(0, m.channels());
    write() << "rng2" << "{" << "x" << x << "y" << y;
    if (cn > 0) write() << "cn" << cn;
    write() << "val" << getElem(m, y, x, cn) << "}";
}

static double evalEps(double expected, double actual, double _eps, ERROR_TYPE err)
{
    if (err == ERROR_ABSOLUTE)
        return _eps;
    else if (err == ERROR_RELATIVE)
        return std::max(std::abs(expected), std::abs(actual)) * err;
    return 0;
}

void Regression::verify(cv::FileNode node, cv::Mat actual, double _eps, std::string argname, ERROR_TYPE err)
{
    double actual_min, actual_max;
    cv::minMaxLoc(actual, &actual_min, &actual_max);

    double eps = evalEps((double)node["min"], actual_min, _eps, err);
    ASSERT_NEAR((double)node["min"], actual_min, eps)
            << "  " << argname << " has unexpected minimal value";

    eps = evalEps((double)node["max"], actual_max, _eps, err);
    ASSERT_NEAR((double)node["max"], actual_max, eps)
            << "  " << argname << " has unexpected maximal value";

    cv::FileNode last = node["last"];
    double actualLast = getElem(actual, actual.rows - 1, actual.cols - 1, actual.channels() - 1);
    ASSERT_EQ((int)last["x"], actual.cols - 1)
            << "  " << argname << " has unexpected number of columns";
    ASSERT_EQ((int)last["y"], actual.rows - 1)
            << "  " << argname << " has unexpected number of rows";

    eps = evalEps((double)last["val"], actualLast, _eps, err);
    ASSERT_NEAR((double)last["val"], actualLast, eps)
            << "  " << argname << " has unexpected value of last element";

    cv::FileNode rng1 = node["rng1"];
    int x1 = rng1["x"];
    int y1 = rng1["y"];
    int cn1 = rng1["cn"];

    eps = evalEps((double)rng1["val"], getElem(actual, y1, x1, cn1), _eps, err);
    ASSERT_NEAR((double)rng1["val"], getElem(actual, y1, x1, cn1), eps)
            << "  " << argname << " has unexpected value of ["<< x1 << ":" << y1 << ":" << cn1 <<"] element";

    cv::FileNode rng2 = node["rng2"];
    int x2 = rng2["x"];
    int y2 = rng2["y"];
    int cn2 = rng2["cn"];

    eps = evalEps((double)rng2["val"], getElem(actual, y2, x2, cn2), _eps, err);
    ASSERT_NEAR((double)rng2["val"], getElem(actual, y2, x2, cn2), eps)
            << "  " << argname << " has unexpected value of ["<< x2 << ":" << y2 << ":" << cn2 <<"] element";
}

void Regression::write(cv::InputArray array)
{
    write() << "kind" << array.kind();
    write() << "type" << array.type();
    if (isVector(array))
    {
        int total = (int)array.total();
        int idx = regRNG.uniform(0, total);
        write() << "len" << total;
        write() << "idx" << idx;

        cv::Mat m = array.getMat(idx);

        if (m.total() * m.channels() < 26) //5x5 or smaller
            write() << "val" << m;
        else
            write(m);
    }
    else
    {
        if (array.total() * array.channels() < 26) //5x5 or smaller
            write() << "val" << array.getMat();
        else
            write(array.getMat());
    }
}

static int countViolations(const cv::Mat& expected, const cv::Mat& actual, const cv::Mat& diff, double eps, double* max_violation = 0, double* max_allowed = 0)
{
    cv::Mat diff64f;
    diff.reshape(1).convertTo(diff64f, CV_64F);

    cv::Mat expected_abs = cv::abs(expected.reshape(1));
    cv::Mat actual_abs = cv::abs(actual.reshape(1));
    cv::Mat maximum, mask;
    cv::max(expected_abs, actual_abs, maximum);
    cv::multiply(maximum, cv::Vec<double, 1>(eps), maximum, CV_64F);
    cv::compare(diff64f, maximum, mask, cv::CMP_GT);

    int v = cv::countNonZero(mask);

    if (v > 0 && max_violation != 0 && max_allowed != 0)
    {
        int loc[10];
        cv::minMaxIdx(maximum, 0, max_allowed, 0, loc, mask);
        *max_violation = diff64f.at<double>(loc[1], loc[0]);
    }

    return v;
}

void Regression::verify(cv::FileNode node, cv::InputArray array, double eps, ERROR_TYPE err)
{
    ASSERT_EQ((int)node["kind"], array.kind()) << "  Argument \"" << node.name() << "\" has unexpected kind";
    ASSERT_EQ((int)node["type"], array.type()) << "  Argument \"" << node.name() << "\" has unexpected type";

    cv::FileNode valnode = node["val"];
    if (isVector(array))
    {
        ASSERT_EQ((int)node["len"], (int)array.total()) << "  Vector \"" << node.name() << "\" has unexpected length";
        int idx = node["idx"];

        cv::Mat actual = array.getMat(idx);

        if (valnode.isNone())
        {
            ASSERT_LE((size_t)26, actual.total() * (size_t)actual.channels())
                    << "  \"" << node.name() << "[" <<  idx << "]\" has unexpected number of elements";
            verify(node, actual, eps, cv::format("%s[%d]", node.name().c_str(), idx), err);
        }
        else
        {
            cv::Mat expected;
            valnode >> expected;

            ASSERT_EQ(expected.size(), actual.size())
                    << "  " << node.name() << "[" <<  idx<< "] has unexpected size";

            cv::Mat diff;
            cv::absdiff(expected, actual, diff);

            if (err == ERROR_ABSOLUTE)
            {
                if (!cv::checkRange(diff, true, 0, 0, eps))
                {
                    double max;
                    cv::minMaxLoc(diff.reshape(1), 0, &max);
                    FAIL() << "  Absolute difference (=" << max << ") between argument \""
                           << node.name() << "[" <<  idx << "]\" and expected value is bugger than " << eps;
                }
            }
            else if (err == ERROR_RELATIVE)
            {
                double maxv, maxa;
                int violations = countViolations(expected, actual, diff, eps, &maxv, &maxa);
                if (violations > 0)
                {
                    FAIL() << "  Relative difference (" << maxv << " of " << maxa << " allowed) between argument \""
                           << node.name() << "[" <<  idx << "]\" and expected value is bugger than " << eps << " in " << violations << " points";
                }
            }
        }
    }
    else
    {
        if (valnode.isNone())
        {
            ASSERT_LE((size_t)26, array.total() * (size_t)array.channels())
                    << "  Argument \"" << node.name() << "\" has unexpected number of elements";
            verify(node, array.getMat(), eps, "Argument " + node.name(), err);
        }
        else
        {
            cv::Mat expected;
            valnode >> expected;
            cv::Mat actual = array.getMat();

            ASSERT_EQ(expected.size(), actual.size())
                    << "  Argument \"" << node.name() << "\" has unexpected size";

            cv::Mat diff;
            cv::absdiff(expected, actual, diff);

            if (err == ERROR_ABSOLUTE)
            {
                if (!cv::checkRange(diff, true, 0, 0, eps))
                {
                    double max;
                    cv::minMaxLoc(diff.reshape(1), 0, &max);
                    FAIL() << "  Difference (=" << max << ") between argument \"" << node.name()
                           << "\" and expected value is bugger than " << eps;
                }
            }
            else if (err == ERROR_RELATIVE)
            {
                double maxv, maxa;
                int violations = countViolations(expected, actual, diff, eps, &maxv, &maxa);
                if (violations > 0)
                {
                    FAIL() << "  Relative difference (" << maxv << " of " << maxa << " allowed) between argument \"" << node.name()
                           << "\" and expected value is bugger than " << eps << " in " << violations << " points";
                }
            }
        }
    }
}

Regression& Regression::operator() (const std::string& name, cv::InputArray array, double eps, ERROR_TYPE err)
{
    std::string nodename = getCurrentTestNodeName();

    cv::FileNode n = rootIn[nodename];
    if(n.isNone())
    {
        if(param_write_sanity)
        {
            if (nodename != currentTestNodeName)
            {
                if (!currentTestNodeName.empty())
                    write() << "}";
                currentTestNodeName = nodename;

                write() << nodename << "{";
            }
            write() << name << "{";
            write(array);
            write() << "}";
        }
    }
    else
    {
        cv::FileNode this_arg = n[name];
        if (!this_arg.isMap())
            ADD_FAILURE() << "  No regression data for " << name << " argument";
        else
            verify(this_arg, array, eps, err);
    }
    return *this;
}


/*****************************************************************************************\
*                                ::perf::performance_metrics
\*****************************************************************************************/
performance_metrics::performance_metrics()
{
    bytesIn = 0;
    bytesOut = 0;
    samples = 0;
    outliers = 0;
    gmean = 0;
    gstddev = 0;
    mean = 0;
    stddev = 0;
    median = 0;
    min = 0;
    frequency = 0;
    terminationReason = TERM_UNKNOWN;
}


/*****************************************************************************************\
*                                   ::perf::TestBase
\*****************************************************************************************/
void TestBase::Init(int argc, const char* const argv[])
{
    cv::CommandLineParser args(argc, argv, command_line_keys);
    param_max_outliers = std::min(100., std::max(0., args.get<double>("perf_max_outliers")));
    param_min_samples  = std::max(1u, args.get<unsigned int>("perf_min_samples"));
    param_max_deviation = std::max(0., args.get<double>("perf_max_deviation"));
    param_seed = args.get<uint64>("perf_seed");
    param_time_limit = std::max(0., args.get<double>("perf_time_limit"));
    param_force_samples = args.get<unsigned int>("perf_force_samples");
    param_write_sanity = args.get<bool>("perf_write_sanity");

    param_tbb_nthreads  = args.get<int>("perf_tbb_nthreads");
#if ANDROID
    param_affinity_mask = args.get<int>("perf_affinity_mask");
#endif

    if (args.get<bool>("help"))
    {
        args.printParams();
        printf("\n\n");
        return;
    }

    timeLimitDefault = param_time_limit == 0.0 ? 1 : (int64)(param_time_limit * cv::getTickFrequency());
    iterationsLimitDefault = param_force_samples == 0 ? (unsigned)(-1) : param_force_samples;
    _timeadjustment = _calibrate();
}

int64 TestBase::_calibrate()
{
    class _helper : public ::perf::TestBase
    {
        public:
        performance_metrics& getMetrics() { return calcMetrics(); }
        virtual void TestBody() {}
        virtual void PerfTestBody()
        {
            //the whole system warmup
            SetUp();
            cv::Mat a(2048, 2048, CV_32S, cv::Scalar(1));
            cv::Mat b(2048, 2048, CV_32S, cv::Scalar(2));
            declare.time(30);
            double s = 0;
            for(declare.iterations(20); startTimer(), next(); stopTimer())
                s+=a.dot(b);
            declare.time(s);

            //self calibration
            SetUp();
            for(declare.iterations(1000); startTimer(), next(); stopTimer()){}
        }
    };

    _timeadjustment = 0;
    _helper h;
    h.PerfTestBody();
    double compensation = h.getMetrics().min;
    LOGD("Time compensation is %.0f", compensation);
    return (int64)compensation;
}

TestBase::TestBase(): declare(this)
{
}

void TestBase::declareArray(SizeVector& sizes, cv::InputOutputArray a, int wtype)
{
    if (!a.empty())
    {
        sizes.push_back(std::pair<int, cv::Size>(getSizeInBytes(a), getSize(a)));
        warmup(a, wtype);
    }
    else if (a.kind() != cv::_InputArray::NONE)
        ADD_FAILURE() << "  Uninitialized input/output parameters are not allowed for performance tests";
}

void TestBase::warmup(cv::InputOutputArray a, int wtype)
{
    if (a.empty()) return;
    if (a.kind() != cv::_InputArray::STD_VECTOR_MAT && a.kind() != cv::_InputArray::STD_VECTOR_VECTOR)
        warmup_impl(a.getMat(), wtype);
    else
    {
        size_t total = a.total();
        for (size_t i = 0; i < total; ++i)
            warmup_impl(a.getMat((int)i), wtype);
    }
}

int TestBase::getSizeInBytes(cv::InputArray a)
{
    if (a.empty()) return 0;
    int total = (int)a.total();
    if (a.kind() != cv::_InputArray::STD_VECTOR_MAT && a.kind() != cv::_InputArray::STD_VECTOR_VECTOR)
        return total * CV_ELEM_SIZE(a.type());

    int size = 0;
    for (int i = 0; i < total; ++i)
        size += (int)a.total(i) * CV_ELEM_SIZE(a.type(i));

    return size;
}

cv::Size TestBase::getSize(cv::InputArray a)
{
    if (a.kind() != cv::_InputArray::STD_VECTOR_MAT && a.kind() != cv::_InputArray::STD_VECTOR_VECTOR)
        return a.size();
    return cv::Size();
}

bool TestBase::next()
{
    return ++currentIter < nIters && totalTime < timeLimit;
}

void TestBase::warmup_impl(cv::Mat m, int wtype)
{
    switch(wtype)
    {
    case WARMUP_READ:
        cv::sum(m.reshape(1));
        return;
    case WARMUP_WRITE:
        m.reshape(1).setTo(cv::Scalar::all(0));
        return;
    case WARMUP_RNG:
        randu(m);
        return;
    default:
        return;
    }
}

unsigned int TestBase::getTotalInputSize() const
{
    unsigned int res = 0;
    for (SizeVector::const_iterator i = inputData.begin(); i != inputData.end(); ++i)
        res += i->first;
    return res;
}

unsigned int TestBase::getTotalOutputSize() const
{
    unsigned int res = 0;
    for (SizeVector::const_iterator i = outputData.begin(); i != outputData.end(); ++i)
        res += i->first;
    return res;
}

void TestBase::startTimer()
{
    lastTime = cv::getTickCount();
}

void TestBase::stopTimer()
{
    int64 time = cv::getTickCount();
    if (lastTime == 0)
        ADD_FAILURE() << "  stopTimer() is called before startTimer()";
    lastTime = time - lastTime;
    totalTime += lastTime;
    lastTime -= _timeadjustment;
    if (lastTime < 0) lastTime = 0;
    times.push_back(lastTime);
    lastTime = 0;
}

performance_metrics& TestBase::calcMetrics()
{
    if ((metrics.samples == (unsigned int)currentIter) || times.size() == 0)
        return metrics;

    metrics.bytesIn = getTotalInputSize();
    metrics.bytesOut = getTotalOutputSize();
    metrics.frequency = cv::getTickFrequency();
    metrics.samples = (unsigned int)times.size();
    metrics.outliers = 0;

    if (metrics.terminationReason != performance_metrics::TERM_INTERRUPT && metrics.terminationReason != performance_metrics::TERM_EXCEPTION)
    {
        if (currentIter == nIters)
            metrics.terminationReason = performance_metrics::TERM_ITERATIONS;
        else if (totalTime >= timeLimit)
            metrics.terminationReason = performance_metrics::TERM_TIME;
        else
            metrics.terminationReason = performance_metrics::TERM_UNKNOWN;
    }

    std::sort(times.begin(), times.end());

    //estimate mean and stddev for log(time)
    double gmean = 0;
    double gstddev = 0;
    int n = 0;
    for(TimeVector::const_iterator i = times.begin(); i != times.end(); ++i)
    {
        double x = (double)*i;
        if (x < DBL_EPSILON) continue;
        double lx = log(x);

        ++n;
        double delta = lx - gmean;
        gmean += delta / n;
        gstddev += delta * (lx - gmean);
    }

    gstddev = n > 1 ? sqrt(gstddev / (n - 1)) : 0;

    TimeVector::const_iterator start = times.begin();
    TimeVector::const_iterator end = times.end();

    //filter outliers assuming log-normal distribution
    //http://stackoverflow.com/questions/1867426/modeling-distribution-of-performance-measurements
    int offset = 0;
    if (gstddev > DBL_EPSILON)
    {
        double minout = exp(gmean - 3 * gstddev);
        double maxout = exp(gmean + 3 * gstddev);
        while(*start < minout) ++start, ++metrics.outliers, ++offset;
        do --end, ++metrics.outliers; while(*end > maxout);
        ++end, --metrics.outliers;
    }

    metrics.min = (double)*start;
    //calc final metrics
    n = 0;
    gmean = 0;
    gstddev = 0;
    double mean = 0;
    double stddev = 0;
    int m = 0;
    for(; start != end; ++start)
    {
        double x = (double)*start;
        if (x > DBL_EPSILON)
        {
            double lx = log(x);
            ++m;
            double gdelta = lx - gmean;
            gmean += gdelta / m;
            gstddev += gdelta * (lx - gmean);
        }
        ++n;
        double delta = x - mean;
        mean += delta / n;
        stddev += delta * (x - mean);
    }

    metrics.mean = mean;
    metrics.gmean = exp(gmean);
    metrics.gstddev = m > 1 ? sqrt(gstddev / (m - 1)) : 0;
    metrics.stddev = n > 1 ? sqrt(stddev / (n - 1)) : 0;
    metrics.median = n % 2
            ? (double)times[offset + n / 2]
            : 0.5 * (times[offset + n / 2] + times[offset + n / 2 - 1]);

    return metrics;
}

void TestBase::validateMetrics()
{
    performance_metrics& m = calcMetrics();

    if (HasFailure()) return;

    ASSERT_GE(m.samples, 1u)
      << "  No time measurements was performed.\nstartTimer() and stopTimer() commands are required for performance tests.";

    EXPECT_GE(m.samples, param_min_samples)
      << "  Only a few samples are collected.\nPlease increase number of iterations or/and time limit to get reliable performance measurements.";

    if (m.gstddev > DBL_EPSILON)
    {
        EXPECT_GT(/*m.gmean * */1., /*m.gmean * */ 2 * sinh(m.gstddev * param_max_deviation))
          << "  Test results are not reliable ((mean-sigma,mean+sigma) deviation interval is bigger than measured time interval).";
    }

    EXPECT_LE(m.outliers, std::max((unsigned int)cvCeil(m.samples * param_max_outliers / 100.), 1u))
      << "  Test results are not reliable (too many outliers).";
}

void TestBase::reportMetrics(bool toJUnitXML)
{
    performance_metrics& m = calcMetrics();

    if (toJUnitXML)
    {
        RecordProperty("bytesIn", (int)m.bytesIn);
        RecordProperty("bytesOut", (int)m.bytesOut);
        RecordProperty("term", m.terminationReason);
        RecordProperty("samples", (int)m.samples);
        RecordProperty("outliers", (int)m.outliers);
        RecordProperty("frequency", cv::format("%.0f", m.frequency).c_str());
        RecordProperty("min", cv::format("%.0f", m.min).c_str());
        RecordProperty("median", cv::format("%.0f", m.median).c_str());
        RecordProperty("gmean", cv::format("%.0f", m.gmean).c_str());
        RecordProperty("gstddev", cv::format("%.6f", m.gstddev).c_str());
        RecordProperty("mean", cv::format("%.0f", m.mean).c_str());
        RecordProperty("stddev", cv::format("%.0f", m.stddev).c_str());
    }
    else
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        const char* type_param = test_info->type_param();
        const char* value_param = test_info->value_param();

#if defined(ANDROID) && defined(USE_ANDROID_LOGGING)
        LOGD("[ FAILED   ] %s.%s", test_info->test_case_name(), test_info->name());
#endif

        if (type_param)  LOGD("type      = %11s", type_param);
        if (value_param) LOGD("params    = %11s", value_param);

        switch (m.terminationReason)
        {
        case performance_metrics::TERM_ITERATIONS:
            LOGD("termination reason:  reached maximum number of iterations");
            break;
        case performance_metrics::TERM_TIME:
            LOGD("termination reason:  reached time limit");
            break;
        case performance_metrics::TERM_INTERRUPT:
            LOGD("termination reason:  aborted by the performance testing framework");
            break;
        case performance_metrics::TERM_EXCEPTION:
            LOGD("termination reason:  unhandled exception");
            break;
        case performance_metrics::TERM_UNKNOWN:
        default:
            LOGD("termination reason:  unknown");
            break;
        };

        LOGD("bytesIn   =%11lu", (unsigned long)m.bytesIn);
        LOGD("bytesOut  =%11lu", (unsigned long)m.bytesOut);
        if (nIters == (unsigned int)-1 || m.terminationReason == performance_metrics::TERM_ITERATIONS)
            LOGD("samples   =%11u",  m.samples);
        else
            LOGD("samples   =%11u of %u", m.samples, nIters);
        LOGD("outliers  =%11u", m.outliers);
        LOGD("frequency =%11.0f", m.frequency);
        if (m.samples > 0)
        {
            LOGD("min       =%11.0f = %.2fms", m.min, m.min * 1e3 / m.frequency);
            LOGD("median    =%11.0f = %.2fms", m.median, m.median * 1e3 / m.frequency);
            LOGD("gmean     =%11.0f = %.2fms", m.gmean, m.gmean * 1e3 / m.frequency);
            LOGD("gstddev   =%11.8f = %.2fms for 97%% dispersion interval", m.gstddev, m.gmean * 2 * sinh(m.gstddev * 3) * 1e3 / m.frequency);
            LOGD("mean      =%11.0f = %.2fms", m.mean, m.mean * 1e3 / m.frequency);
            LOGD("stddev    =%11.0f = %.2fms", m.stddev, m.stddev * 1e3 / m.frequency);
        }
    }
}

void TestBase::SetUp()
{
#ifdef HAVE_TBB
    if (param_tbb_nthreads > 0) {
        p_tbb_initializer.release();
        p_tbb_initializer=new tbb::task_scheduler_init(param_tbb_nthreads);
    }
#endif
#if ANDROID
    if (param_affinity_mask)
        setCurrentThreadAffinityMask(param_affinity_mask);
#endif
    lastTime = 0;
    totalTime = 0;
    nIters = iterationsLimitDefault;
    currentIter = (unsigned int)-1;
    timeLimit = timeLimitDefault;
    times.clear();
    cv::theRNG().state = param_seed;//this rng should generate same numbers for each run
}

void TestBase::TearDown()
{
    validateMetrics();
    if (HasFailure())
        reportMetrics(false);
    else
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        const char* type_param = test_info->type_param();
        const char* value_param = test_info->value_param();
        if (value_param) printf("[ VALUE    ] \t%s\n", value_param), fflush(stdout);
        if (type_param)  printf("[ TYPE     ] \t%s\n", type_param), fflush(stdout);
        reportMetrics(true);
    }
#ifdef HAVE_TBB
    p_tbb_initializer.release();
#endif
}

std::string TestBase::getDataPath(const std::string& relativePath)
{
    if (relativePath.empty())
    {
        ADD_FAILURE() << "  Bad path to test resource";
        throw PerfEarlyExitException();
    }

    const char *data_path_dir = getenv("OPENCV_TEST_DATA_PATH");
    const char *path_separator = "/";

    std::string path;
    if (data_path_dir)
    {
        int len = (int)strlen(data_path_dir) - 1;
        if (len < 0) len = 0;
        path = (data_path_dir[0] == 0 ? std::string(".") : std::string(data_path_dir))
                + (data_path_dir[len] == '/' || data_path_dir[len] == '\\' ? "" : path_separator);
    }
    else
    {
        path = ".";
        path += path_separator;
    }

    if (relativePath[0] == '/' || relativePath[0] == '\\')
        path += relativePath.substr(1);
    else
        path += relativePath;

    FILE* fp = fopen(path.c_str(), "r");
    if (fp)
        fclose(fp);
    else
    {
        ADD_FAILURE() << "  Requested file \"" << path << "\" does not exist.";
        throw PerfEarlyExitException();
    }
    return path;
}

void TestBase::RunPerfTestBody()
{
    try
    {
        this->PerfTestBody();
    }
    catch(PerfEarlyExitException)
    {
        metrics.terminationReason = performance_metrics::TERM_INTERRUPT;
        return;//no additional failure logging
    }
    catch(cv::Exception e)
    {
        metrics.terminationReason = performance_metrics::TERM_EXCEPTION;
        FAIL() << "Expected: PerfTestBody() doesn't throw an exception.\n  Actual: it throws:\n  " << e.what();
    }
    catch(...)
    {
        metrics.terminationReason = performance_metrics::TERM_EXCEPTION;
        FAIL() << "Expected: PerfTestBody() doesn't throw an exception.\n  Actual: it throws.";
    }
}

/*****************************************************************************************\
*                          ::perf::TestBase::_declareHelper
\*****************************************************************************************/
TestBase::_declareHelper& TestBase::_declareHelper::iterations(unsigned int n)
{
    test->times.clear();
    test->times.reserve(n);
    test->nIters = std::min(n, TestBase::iterationsLimitDefault);
    test->currentIter = (unsigned int)-1;
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::time(double timeLimitSecs)
{
    test->times.clear();
    test->currentIter = (unsigned int)-1;
    test->timeLimit = (int64)(timeLimitSecs * cv::getTickFrequency());
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::tbb_threads(int n)
{
#ifdef HAVE_TBB
    test->p_tbb_initializer.release();
    if (n > 0)
        test->p_tbb_initializer=new tbb::task_scheduler_init(n);
#endif
	(void)n;
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::in(cv::InputOutputArray a1, int wtype)
{
    if (!test->times.empty()) return *this;
    TestBase::declareArray(test->inputData, a1, wtype);
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::in(cv::InputOutputArray a1, cv::InputOutputArray a2, int wtype)
{
    if (!test->times.empty()) return *this;
    TestBase::declareArray(test->inputData, a1, wtype);
    TestBase::declareArray(test->inputData, a2, wtype);
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::in(cv::InputOutputArray a1, cv::InputOutputArray a2, cv::InputOutputArray a3, int wtype)
{
    if (!test->times.empty()) return *this;
    TestBase::declareArray(test->inputData, a1, wtype);
    TestBase::declareArray(test->inputData, a2, wtype);
    TestBase::declareArray(test->inputData, a3, wtype);
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::in(cv::InputOutputArray a1, cv::InputOutputArray a2, cv::InputOutputArray a3, cv::InputOutputArray a4, int wtype)
{
    if (!test->times.empty()) return *this;
    TestBase::declareArray(test->inputData, a1, wtype);
    TestBase::declareArray(test->inputData, a2, wtype);
    TestBase::declareArray(test->inputData, a3, wtype);
    TestBase::declareArray(test->inputData, a4, wtype);
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::out(cv::InputOutputArray a1, int wtype)
{
    if (!test->times.empty()) return *this;
    TestBase::declareArray(test->outputData, a1, wtype);
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::out(cv::InputOutputArray a1, cv::InputOutputArray a2, int wtype)
{
    if (!test->times.empty()) return *this;
    TestBase::declareArray(test->outputData, a1, wtype);
    TestBase::declareArray(test->outputData, a2, wtype);
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::out(cv::InputOutputArray a1, cv::InputOutputArray a2, cv::InputOutputArray a3, int wtype)
{
    if (!test->times.empty()) return *this;
    TestBase::declareArray(test->outputData, a1, wtype);
    TestBase::declareArray(test->outputData, a2, wtype);
    TestBase::declareArray(test->outputData, a3, wtype);
    return *this;
}

TestBase::_declareHelper& TestBase::_declareHelper::out(cv::InputOutputArray a1, cv::InputOutputArray a2, cv::InputOutputArray a3, cv::InputOutputArray a4, int wtype)
{
    if (!test->times.empty()) return *this;
    TestBase::declareArray(test->outputData, a1, wtype);
    TestBase::declareArray(test->outputData, a2, wtype);
    TestBase::declareArray(test->outputData, a3, wtype);
    TestBase::declareArray(test->outputData, a4, wtype);
    return *this;
}

TestBase::_declareHelper::_declareHelper(TestBase* t) : test(t)
{
}

/*****************************************************************************************\
*                                  ::perf::PrintTo
\*****************************************************************************************/
namespace perf
{

void PrintTo(const MatType& t, ::std::ostream* os)
{
    switch( CV_MAT_DEPTH((int)t) )
    {
        case CV_8U:  *os << "8U";  break;
        case CV_8S:  *os << "8S";  break;
        case CV_16U: *os << "16U"; break;
        case CV_16S: *os << "16S"; break;
        case CV_32S: *os << "32S"; break;
        case CV_32F: *os << "32F"; break;
        case CV_64F: *os << "64F"; break;
        case CV_USRTYPE1: *os << "USRTYPE1"; break;
        default: *os << "INVALID_TYPE"; break;
    }
    *os << 'C' << CV_MAT_CN((int)t);
}

} //namespace perf

/*****************************************************************************************\
*                                  ::cv::PrintTo
\*****************************************************************************************/
namespace cv {

void PrintTo(const Size& sz, ::std::ostream* os)
{
    *os << /*"Size:" << */sz.width << "x" << sz.height;
}

}  // namespace cv


/*****************************************************************************************\
*                                  ::cv::PrintTo
\*****************************************************************************************/
