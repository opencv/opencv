// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

#if defined(HAVE_JPEG) || defined(HAVE_PNG)
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && defined(TARGET_OS_OSX) && TARGET_OS_OSX)
#define OPENCV_TEST_IMWRITE_SPAWN 1
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace opencv_test { namespace {

static Mat makeWriteTestImage(int side)
{
    Mat image(side, side, CV_8UC3);
    RNG(0x12345678).fill(image, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
    return image;
}

struct WriteTestFile
{
    explicit WriteTestFile(const String& extension) : name(cv::tempfile(extension.c_str())) {}
    ~WriteTestFile() { remove(name.c_str()); }
    String name;
};

typedef testing::TestWithParam<tuple<String, int> > Imgcodecs_ImwriteErrors;

TEST_P(Imgcodecs_ImwriteErrors, file_matches_memory)
{
    const String extension = get<0>(GetParam());
    const Mat image = makeWriteTestImage(get<1>(GetParam()));
    std::vector<uchar> encoded, written;
    ASSERT_TRUE(imencode(extension, image, encoded));
    const WriteTestFile file(extension);
    ASSERT_TRUE(imwrite(file.name, image));
    readFileBytes(file.name, written);
    EXPECT_EQ(encoded, written);
}

#ifdef OPENCV_TEST_IMWRITE_SPAWN
// This helper is explicitly selected in a fresh process. The distinct success
// exit code proves it ran, rather than being skipped by test filtering/sharding.
TEST(Imgcodecs_ImwriteErrorsChild, DISABLED_file_size_limit)
{
    const char* extension = std::getenv("OPENCV_IMWRITE_ERROR_TEST_EXT");
    const char* side = std::getenv("OPENCV_IMWRITE_ERROR_TEST_SIDE");
    if (!extension || !side)
        throw SkipTestException("Run by the file_size_limit parent test only");
    Mat image;
    std::vector<uchar> encoded;
    try
    {
        image = makeWriteTestImage(std::atoi(side));
        if (!imencode(extension, image, encoded) || encoded.size() < 2)
            _exit(2);
    }
    catch (...) { _exit(2); }

    struct rlimit limit;
    if (getrlimit(RLIMIT_FSIZE, &limit) != 0)
        _exit(2);
    limit.rlim_cur = std::min(limit.rlim_max,
            static_cast<rlim_t>(std::min<size_t>(1024, encoded.size() / 2)));
    if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR || setrlimit(RLIMIT_FSIZE, &limit) != 0)
        _exit(2);

    String filename;
    try { filename = cv::tempfile(extension); }
    catch (...) { _exit(2); }
    int status;
    try { status = imwrite(filename, image) ? 1 : 42; }
    catch (...) { status = 3; }
    remove(filename.c_str());
    _exit(status);
}

TEST_P(Imgcodecs_ImwriteErrors, file_size_limit)
{
    const std::vector<std::string> originalArguments = testing::internal::GetArgvs();
    ASSERT_FALSE(originalArguments.empty());
    std::vector<std::string> arguments = {
        originalArguments[0],
        "--gtest_filter=Imgcodecs_ImwriteErrorsChild.DISABLED_file_size_limit",
        "--gtest_also_run_disabled_tests", "--gtest_repeat=1", "--gtest_output="
    };
    std::vector<char*> argv;
    for (size_t i = 0; i < arguments.size(); ++i)
        argv.push_back(const_cast<char*>(arguments[i].c_str()));
    argv.push_back(NULL);

    std::vector<std::string> environment;
    for (char** entry = environ; entry && *entry; ++entry)
    {
        const std::string value(*entry);
        if (value.find("OPENCV_IMWRITE_ERROR_TEST_") == 0 ||
            value.find("GTEST_TOTAL_SHARDS=") == 0 ||
            value.find("GTEST_SHARD_INDEX=") == 0 ||
            value.find("GTEST_SHARD_STATUS_FILE=") == 0)
            continue;
        environment.push_back(value);
    }
    environment.push_back("OPENCV_IMWRITE_ERROR_TEST_EXT=" + get<0>(GetParam()));
    environment.push_back(format("OPENCV_IMWRITE_ERROR_TEST_SIDE=%d", get<1>(GetParam())));
    std::vector<char*> envp;
    for (size_t i = 0; i < environment.size(); ++i)
        envp.push_back(const_cast<char*>(environment[i].c_str()));
    envp.push_back(NULL);

    // posix_spawn re-executes safely even when earlier tests started threads.
    // It does not depend on gtest's optional saved working directory.
    posix_spawn_file_actions_t actions;
    ASSERT_EQ(0, posix_spawn_file_actions_init(&actions));
    const int redirectResult = posix_spawn_file_actions_addopen(
            &actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t child = -1;
    const int spawnResult = redirectResult == 0
            ? posix_spawnp(&child, argv[0], &actions, NULL, argv.data(), envp.data())
            : redirectResult;
    posix_spawn_file_actions_destroy(&actions);
    ASSERT_EQ(0, spawnResult);
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); }
    while (waited == -1 && errno == EINTR);
    ASSERT_EQ(child, waited);
    ASSERT_TRUE(WIFEXITED(status)) << "Child status: " << status;
    EXPECT_EQ(42, WEXITSTATUS(status)) << "1: write succeeded; 2: setup failed; 3: write threw";
}
#endif

const String writeErrorExtensions[] = {
#ifdef HAVE_JPEG
    ".jpg",
#endif
#ifdef HAVE_PNG
    ".png",
#endif
};
INSTANTIATE_TEST_CASE_P(All, Imgcodecs_ImwriteErrors,
        testing::Combine(testing::ValuesIn(writeErrorExtensions), testing::Values(32, 512)));

}} // namespace
#endif // HAVE_JPEG || HAVE_PNG
