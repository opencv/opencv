// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/base.hpp"
#include "test_precomp.hpp"
#include <limits>

namespace opencv_test {
namespace {

TEST(Core_TermCriteria, EmptyConstructorIsNotValid)
{
    TermCriteria termCriteria;

    EXPECT_FALSE(termCriteria.isValid());
}

TEST(Core_TermCriteria, FullConstructorIsValid)
{
    TermCriteria termCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 0.1);

    EXPECT_TRUE(termCriteria.isValid());
}

TEST(Core_TermCriteria, MaxIterConstructorIsValid)
{
    TermCriteria termCriteria(TermCriteria::MAX_ITER, 10, 0);

    EXPECT_TRUE(termCriteria.isValid());
}

TEST(Core_TermCriteria, EpsilonConstructorIsValid)
{
    TermCriteria termCriteria(TermCriteria::EPS, 0, 0.1);

    EXPECT_TRUE(termCriteria.isValid());
}

TEST(Core_TermCriteria, IncompleteMaxCountConstructorIsNotValid)
{
    TermCriteria termCriteria(0, 5, 0.1);

    EXPECT_FALSE(termCriteria.isValid());
}

TEST(Core_TermCriteria, IncompletEpsilonConstructorIsNotValid)
{
    TermCriteria termCriteria(TermCriteria::MAX_ITER, 0, 0.1);

    EXPECT_FALSE(termCriteria.isValid());
}

TEST(Core_TermCriteria, c)
{
    TermCriteria termCriteria(5, 0.1);

    EXPECT_TRUE(termCriteria.isEpsilonToleranceSet());

    EXPECT_EQ(termCriteria.getEpsilonTolerance(), 0.1);
    EXPECT_EQ(termCriteria.getIterationTolerance(), 5);
}

TEST(Core_TermCriteria, ToleranceCheckers)
{
    TermCriteria termCriteria(5, 0.1);

    EXPECT_TRUE(termCriteria.checkIterationTolerance(4));
    EXPECT_TRUE(termCriteria.checkIterationTolerance(5));
    EXPECT_FALSE(termCriteria.checkIterationTolerance(6));

    EXPECT_TRUE(termCriteria.checkEpsilonTolerance(0.005));
    EXPECT_FALSE(termCriteria.checkEpsilonTolerance(0.2));
}

TEST(Core_TermCriteria, GettersSetters)
{
    TermCriteria termCriteria;

    termCriteria.setEpsilonTolerance(0.1);
    termCriteria.setIterationTolerance(5);

    EXPECT_EQ(termCriteria.getEpsilonTolerance(), 0.1);
    EXPECT_EQ(termCriteria.getIterationTolerance(), 5);

    termCriteria.setIterationTolerance(-1);
    EXPECT_EQ(termCriteria.getIterationTolerance(), std::numeric_limits<int>::max());
}

TEST(Core_termCriteria, NewOld)
{
    TermCriteria termCriteria(5, 0.1);

    EXPECT_TRUE(termCriteria.isValid());
    EXPECT_EQ(termCriteria.type, TermCriteria::MAX_ITER + TermCriteria::EPS);
    EXPECT_EQ(termCriteria.epsilon, 0.1);
    EXPECT_EQ(termCriteria.maxCount, 5);
}

TEST(Core_termCriteria, NewOldAllDisabled)
{
    TermCriteria termCriteria(-1, -1);

    EXPECT_TRUE(termCriteria.isValid());
    EXPECT_EQ(termCriteria.type, TermCriteria::MAX_ITER);
    EXPECT_EQ(termCriteria.maxCount, std::numeric_limits<int>::max());

    termCriteria.setEpsilonTolerance(0.1);
    EXPECT_TRUE(termCriteria.isValid());
    EXPECT_EQ(termCriteria.type, TermCriteria::MAX_ITER + TermCriteria::EPS);
    EXPECT_EQ(termCriteria.maxCount, std::numeric_limits<int>::max());
    EXPECT_EQ(termCriteria.epsilon, 0.1);
}

} // namespace
} // namespace opencv_test
