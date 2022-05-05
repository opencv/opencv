#include "opencv2/core/base.hpp"
#include "test_precomp.hpp"

namespace opencv_test { namespace {

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

}} // namespace opencv_test, namespace
