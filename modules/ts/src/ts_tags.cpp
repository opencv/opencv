// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "ts_tags.hpp"

namespace cvtest {

static bool printTestTag = false;

static std::vector<std::string> currentDirectTestTags, currentImpliedTestTags;
static std::vector<const ::testing::TestInfo*> skipped_tests;

static std::map<std::string, int>& getTestTagsSkipCounts()
{
    static std::map<std::string, int> testTagsSkipCounts;
    return testTagsSkipCounts;
}
static std::map<std::string, int>& getTestTagsSkipExtraCounts()
{
    static std::map<std::string, int> testTagsSkipExtraCounts;
    return testTagsSkipExtraCounts;
}
void testTagIncreaseSkipCount(const std::string& tag, bool isMain, bool appendSkipTests)
{
    if (appendSkipTests)
        skipped_tests.push_back(::testing::UnitTest::GetInstance()->current_test_info());
    std::map<std::string, int>& counts = isMain ? getTestTagsSkipCounts() : getTestTagsSkipExtraCounts();
    std::map<std::string, int>::iterator i = counts.find(tag);
    if (i == counts.end())
    {
        counts[tag] = 1;
    }
    else
    {
        i->second++;
    }
}

static std::vector<std::string>& getTestTagsSkipList()
{
    static std::vector<std::string> testSkipWithTags;
    static bool initialized = false;
    if (!initialized)
    {
#if OPENCV_32BIT_CONFIGURATION
        testSkipWithTags.push_back(CV_TEST_TAG_MEMORY_2GB);
#else
        if (!cvtest::runBigDataTests)
            testSkipWithTags.push_back(CV_TEST_TAG_MEMORY_6GB);
#endif
        testSkipWithTags.push_back(CV_TEST_TAG_VERYLONG);
#if defined(_DEBUG)
        testSkipWithTags.push_back(CV_TEST_TAG_DEBUG_VERYLONG);
#endif
        testSkipWithTags.push_back(CV_TEST_TAG_DSP);
        initialized = true;
    }
    return testSkipWithTags;
}

void registerGlobalSkipTag(const std::string& skipTag)
{
    if (skipTag.empty())
        return;  // do nothing
    std::vector<std::string>& skipTags = getTestTagsSkipList();
    for (size_t i = 0; i < skipTags.size(); ++i)
    {
        if (skipTag == skipTags[i])
            return;  // duplicate
    }
    skipTags.push_back(skipTag);
}

static std::vector<std::string>& getTestTagsForceList()
{
    static std::vector<std::string> getTestTagsForceList;
    return getTestTagsForceList;
}

static std::vector<std::string>& getTestTagsRequiredList()
{
    static std::vector<std::string> getTestTagsRequiredList;
    return getTestTagsRequiredList;
}


class TestTagsListener: public ::testing::EmptyTestEventListener
{
public:
    void OnTestProgramStart(const ::testing::UnitTest& /*unit_test*/) CV_OVERRIDE
    {
        {
            const std::vector<std::string>& tags = getTestTagsRequiredList();
            std::ostringstream os, os_direct;
            for (size_t i = 0; i < tags.size(); i++)
            {
                os << (i == 0 ? "'" : ", '") << tags[i] << "'";
                os_direct << (i == 0 ? "" : ",") << tags[i];
            }
            std::string tags_str = os.str();
            if (!tags.empty())
                std::cout << "TEST: Run tests with tags: " << tags_str << std::endl;
            ::testing::Test::RecordProperty("test_tags", os_direct.str());
        }
        {
            const std::vector<std::string>& tags = getTestTagsSkipList();
            std::ostringstream os, os_direct;
            for (size_t i = 0; i < tags.size(); i++)
            {
                os << (i == 0 ? "'" : ", '") << tags[i] << "'";
                os_direct << (i == 0 ? "" : ",") << tags[i];
            }
            std::string tags_str = os.str();
            if (!tags.empty())
                std::cout << "TEST: Skip tests with tags: " << tags_str << std::endl;
            ::testing::Test::RecordProperty("test_tags_skip", os_direct.str());
        }
        {
            const std::vector<std::string>& tags = getTestTagsForceList();
            std::ostringstream os, os_direct;
            for (size_t i = 0; i < tags.size(); i++)
            {
                os << (i == 0 ? "'" : ", '") << tags[i] << "'";
                os_direct << (i == 0 ? "" : ",") << tags[i];
            }
            std::string tags_str = os.str();
            if (!tags.empty())
                std::cout << "TEST: Force tests with tags: " << tags_str << std::endl;
            ::testing::Test::RecordProperty("test_tags_force", os_direct.str());
        }
    }

    void OnTestStart(const ::testing::TestInfo& test_info) CV_OVERRIDE
    {
        currentDirectTestTags.clear();
        currentImpliedTestTags.clear();

        const char* value_param_ = test_info.value_param();
        if (value_param_)
        {
            std::string value_param(value_param_);
            if (value_param.find("CV_64F") != std::string::npos
                || (value_param.find("64F") != std::string::npos
                    && value_param.find(" 64F") != std::string::npos
                    && value_param.find(",64F") != std::string::npos
                    && value_param.find("(64F") != std::string::npos
                )
            )
                applyTestTag_(CV_TEST_TAG_TYPE_64F);
            if (value_param.find("1280x720") != std::string::npos)
                applyTestTag_(CV_TEST_TAG_SIZE_HD);
            if (value_param.find("1920x1080") != std::string::npos)
                applyTestTag_(CV_TEST_TAG_SIZE_FULLHD);
            if (value_param.find("3840x2160") != std::string::npos)
                applyTestTag_(CV_TEST_TAG_SIZE_4K);
        }
    }

    void OnTestEnd(const ::testing::TestInfo& /*test_info*/) CV_OVERRIDE
    {
        if (currentDirectTestTags.empty() && currentImpliedTestTags.empty())
        {
            if (printTestTag) std::cout << "[     TAGS ] No tags" << std::endl;
            return;
        }
        std::ostringstream os;
        std::ostringstream os_direct;
        std::ostringstream os_implied;
        {
            const std::vector<std::string>& tags = currentDirectTestTags;
            for (size_t i = 0; i < tags.size(); i++)
            {
                os << (i == 0 ? "" : ", ") << tags[i];
                os_direct << (i == 0 ? "" : ",") << tags[i];
            }
        }
        if (!currentImpliedTestTags.empty())
        {
            os << " (implied tags: ";
            const std::vector<std::string>& tags = currentImpliedTestTags;
            for (size_t i = 0; i < tags.size(); i++)
            {
                os << (i == 0 ? "" : ", ") << tags[i];
                os_implied << (i == 0 ? "" : ",") << tags[i];
            }
            os << ")";
        }
        if (printTestTag) std::cout << "[     TAGS ] " << os.str() << std::endl;
        ::testing::Test::RecordProperty("tags", os_direct.str());
        ::testing::Test::RecordProperty("tags_implied", os_implied.str());
    }

    void OnTestIterationEnd(const ::testing::UnitTest& /*unit_test*/, int /*iteration*/) CV_OVERRIDE
    {
        if (!skipped_tests.empty())
        {
            std::cout << "[ SKIPSTAT ] " << skipped_tests.size() << " tests skipped" << std::endl;
            const std::vector<std::string>& skipTags = getTestTagsSkipList();
            const std::map<std::string, int>& counts = getTestTagsSkipCounts();
            const std::map<std::string, int>& countsExtra = getTestTagsSkipExtraCounts();
            std::vector<std::string> skipTags_all = skipTags;
            skipTags_all.push_back("skip_bigdata");
            skipTags_all.push_back("skip_other");
            for (std::vector<std::string>::const_iterator i = skipTags_all.begin(); i != skipTags_all.end(); ++i)
            {
                int c1 = 0;
                std::map<std::string, int>::const_iterator i1 = counts.find(*i);
                if (i1 != counts.end()) c1 = i1->second;
                int c2 = 0;
                std::map<std::string, int>::const_iterator i2 = countsExtra.find(*i);
                if (i2 != countsExtra.end()) c2 = i2->second;
                if (c2 > 0)
                {
                    std::cout << "[ SKIPSTAT ] TAG='" << *i << "' skip " << c1 << " tests (" << c2 << " times in extra skip list)" << std::endl;
                }
                else if (c1 > 0)
                {
                    std::cout << "[ SKIPSTAT ] TAG='" << *i << "' skip " << c1 << " tests" << std::endl;
                }
            }
        }
        skipped_tests.clear();
    }

    void OnTestProgramEnd(const ::testing::UnitTest& /*unit_test*/) CV_OVERRIDE
    {
        /*if (!skipped_tests.empty())
        {
            for (size_t i = 0; i < skipped_tests.size(); i++)
            {
                const ::testing::TestInfo* test_info = skipped_tests[i];
                if (!test_info) continue;
                std::cout << "- " << test_info->test_case_name() << "." << test_info->name() << std::endl;
            }
        }*/
    }
};

static bool isTestTagForced(const std::string& testTag)
{
    const std::vector<std::string>& forceTags = getTestTagsForceList();
    for (size_t i = 0; i < forceTags.size(); ++i)
    {
        const std::string& forceTag = forceTags[i];
        if (testTag == forceTag
            || (testTag.size() >= forceTag.size()
                && forceTag[forceTag.size() - 1] == '*'
                && forceTag.substr(0, forceTag.size() - 1) == testTag.substr(0, forceTag.size() - 1)
            )
        )
        {
            return true;
        }
    }
    return false;
}

static bool isTestTagSkipped(const std::string& testTag, CV_OUT std::string& skippedByTag)
{
    skippedByTag.clear();
    const std::vector<std::string>& skipTags = getTestTagsSkipList();
    for (size_t i = 0; i < skipTags.size(); ++i)
    {
        const std::string& skipTag = skipTags[i];
        if (testTag == skipTag
            || (testTag.size() >= skipTag.size()
                && skipTag[skipTag.size() - 1] == '*'
                && skipTag.substr(0, skipTag.size() - 1) == testTag.substr(0, skipTag.size() - 1)
            )
        )
        {
            skippedByTag = skipTag;
            return true;
        }
    }
    return false;
}

void checkTestTags()
{
    std::string skipTag;
    const std::vector<std::string>& testTags = currentDirectTestTags;
    {
        const std::vector<std::string>& tags = getTestTagsRequiredList();
        if (!tags.empty())
        {
            size_t found = 0;
            for (size_t i = 0; i < tags.size(); ++i)
            {
                const std::string& tag = tags[i];
                for (size_t j = 0; j < testTags.size(); ++j)
                {
                    const std::string& testTag = testTags[i];
                    if (testTag == tag
                        || (testTag.size() >= tag.size()
                            && tag[tag.size() - 1] == '*'
                            && tag.substr(0, tag.size() - 1) == testTag.substr(0, tag.size() - 1)
                        )
                    )
                    {
                        found++;
                        break;
                    }
                }
            }
            if (found != tags.size())
            {
                skipped_tests.push_back(::testing::UnitTest::GetInstance()->current_test_info());
                throw details::SkipTestExceptionBase("Test tags don't pass required tags list (--test_tag parameter)", true);
            }
        }
    }
    for (size_t i = 0; i < testTags.size(); ++i)
    {
        const std::string& testTag = testTags[i];
        if (isTestTagForced(testTag))
            return;
    }
    std::string skip_message;
    for (size_t i = 0; i < testTags.size(); ++i)
    {
        const std::string& testTag = testTags[i];
        if (isTestTagSkipped(testTag, skipTag))
        {
            testTagIncreaseSkipCount(skipTag, skip_message.empty());
            if (skip_message.empty()) skip_message = "Test with tag '" + testTag + "' is skipped ('" + skipTag + "' is in skip list)";
        }
    }
    const std::vector<std::string>& testTagsImplied = currentImpliedTestTags;
    for (size_t i = 0; i < testTagsImplied.size(); ++i)
    {
        const std::string& testTag = testTagsImplied[i];
        if (isTestTagSkipped(testTag, skipTag))
        {
            testTagIncreaseSkipCount(skipTag, skip_message.empty());
            if (skip_message.empty()) skip_message = "Test with tag '" + testTag + "' is skipped (implied '" + skipTag + "' is in skip list)";
        }
    }

    if (!skip_message.empty())
    {
        skipped_tests.push_back(::testing::UnitTest::GetInstance()->current_test_info());
        throw details::SkipTestExceptionBase(skip_message, true);
    }
}

static bool applyTestTagImpl(const std::string& tag, bool direct = false)
{
    CV_Assert(!tag.empty());
    std::vector<std::string>& testTags = direct ? currentDirectTestTags : currentImpliedTestTags;
    for (size_t i = 0; i < testTags.size(); ++i)
    {
        const std::string& testTag = testTags[i];
        if (tag == testTag)
        {
            return false;  // already exists, skip
        }
    }
    testTags.push_back(tag);

    // Tags implies logic
    if (tag == CV_TEST_TAG_MEMORY_14GB)
        applyTestTagImpl(CV_TEST_TAG_MEMORY_6GB);
    if (tag == CV_TEST_TAG_MEMORY_6GB)
        applyTestTagImpl(CV_TEST_TAG_MEMORY_2GB);
    if (tag == CV_TEST_TAG_MEMORY_2GB)
        applyTestTagImpl(CV_TEST_TAG_MEMORY_1GB);
    if (tag == CV_TEST_TAG_MEMORY_1GB)
        applyTestTagImpl(CV_TEST_TAG_MEMORY_512MB);
    if (tag == CV_TEST_TAG_VERYLONG)
    {
        applyTestTagImpl(CV_TEST_TAG_DEBUG_VERYLONG);
        applyTestTagImpl(CV_TEST_TAG_LONG);
    }
    else if (tag == CV_TEST_TAG_DEBUG_VERYLONG)
    {
        applyTestTagImpl(CV_TEST_TAG_DEBUG_LONG);
    }
    else if (tag == CV_TEST_TAG_LONG)
    {
        applyTestTagImpl(CV_TEST_TAG_DEBUG_LONG);
    }

    if (tag == CV_TEST_TAG_SIZE_4K)
        applyTestTagImpl(CV_TEST_TAG_SIZE_FULLHD);
    if (tag == CV_TEST_TAG_SIZE_FULLHD)
        applyTestTagImpl(CV_TEST_TAG_SIZE_HD);

    return true;
}

void applyTestTag(const std::string& tag)
{
    if (tag.empty()) return;
    if (!applyTestTagImpl(tag, true))
        return;
    checkTestTags();
}

void applyTestTag_(const std::string& tag)
{
    if (tag.empty()) return;
    if (!applyTestTagImpl(tag, true))
        return;
}

static std::vector<std::string> parseStringList(const std::string& s)
{
    std::vector<std::string> result;
    size_t start_pos = 0;
    while (start_pos != std::string::npos)
    {
        while (start_pos < s.size() && s[start_pos] == ' ')
            start_pos++;
        const size_t pos_ = s.find(',', start_pos);
        size_t pos = (pos_ == std::string::npos ? s.size() : pos_);
        while (pos > start_pos && s[pos - 1] == ' ')
            pos--;
        if (pos > start_pos)
        {
            const std::string one_piece(s, start_pos, pos - start_pos);
            result.push_back(one_piece);
        }
        start_pos = (pos_ == std::string::npos ? pos_ : pos_ + 1);
    }
    return result;

}

void activateTestTags(const cv::CommandLineParser& parser)
{
    std::string test_tag_skip = parser.get<std::string>("test_tag_skip");
    if (!test_tag_skip.empty())
    {
        const std::vector<std::string> tag_list = parseStringList(test_tag_skip);
        if (!tag_list.empty())
        {
            std::vector<std::string>& skipTags = getTestTagsSkipList();
            for (size_t k = 0; k < tag_list.size(); ++k)
            {
                const std::string& tag = tag_list[k];
                bool found = false;
                for (size_t i = 0; i < skipTags.size(); ++i)
                {
                    if (tag == skipTags[i])
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    skipTags.push_back(tag);
            }
        }
    }

    std::string test_tag_enable = parser.get<std::string>("test_tag_enable");
    if (!test_tag_enable.empty())
    {
        const std::vector<std::string> tag_list = parseStringList(test_tag_enable);
        if (!tag_list.empty())
        {
            std::vector<std::string>& skipTags = getTestTagsSkipList();
            for (size_t k = 0; k < tag_list.size(); ++k)
            {
                const std::string& tag = tag_list[k];
                bool found = false;
                for (size_t i = 0; i < skipTags.size(); ++i)
                {
                    if (tag == skipTags[i])
                    {
                        skipTags.erase(skipTags.begin() + i);
                        found = true;
                    }
                }
                if (!found)
                {
                    std::cerr << "Can't re-enable tag '" << tag << "' - it is not in the skip list" << std::endl;
                }
            }
        }
    }

    std::string test_tag_force = parser.get<std::string>("test_tag_force");
    if (!test_tag_force.empty())
    {
        const std::vector<std::string> tag_list = parseStringList(test_tag_force);
        if (!tag_list.empty())
        {
            std::vector<std::string>& forceTags = getTestTagsForceList();
            for (size_t k = 0; k < tag_list.size(); ++k)
            {
                const std::string& tag = tag_list[k];
                bool found = false;
                for (size_t i = 0; i < forceTags.size(); ++i)
                {
                    if (tag == forceTags[i])
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    forceTags.push_back(tag);
            }
        }
    }

    std::string test_tag = parser.get<std::string>("test_tag");
    if (!test_tag.empty())
    {
        const std::vector<std::string> tag_list = parseStringList(test_tag);
        if (!tag_list.empty())
        {
            std::vector<std::string>& requiredTags = getTestTagsRequiredList();
            for (size_t k = 0; k < tag_list.size(); ++k)
            {
                const std::string& tag = tag_list[k];
                bool found = false;
                for (size_t i = 0; i < requiredTags.size(); ++i)
                {
                    if (tag == requiredTags[i])
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    requiredTags.push_back(tag);
            }
        }
    }

    printTestTag = parser.get<bool>("test_tag_print");

    ::testing::UnitTest::GetInstance()->listeners().Append(new TestTagsListener());
}

} // namespace
