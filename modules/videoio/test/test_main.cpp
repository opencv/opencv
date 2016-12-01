#include "test_precomp.hpp"

CV_TEST_MAIN("highgui")


TEST(apiPropInfo, simple)
{
  int prop_id = 10; // some value, it doesn't used actually
  int api = cv::CAP_IMAGES;
  int id = cv::videoio::apiProp(api, prop_id);
  int result_prop = -1;
  int result_api = cv::videoio::apiPropInfo(id, result_prop);

  EXPECT_EQ(prop_id, result_prop);
  EXPECT_EQ(api, result_api);
}
