#include "opencv2/core/base.hpp"
#include "test_precomp.hpp"

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

    EXPECT_EQ(termCriteria.getIterationTolerance(), 0.1);
    EXPECT_EQ(termCriteria.getIterationTolerance(), 5);
}

} // namespace
} // namespace opencv_test
