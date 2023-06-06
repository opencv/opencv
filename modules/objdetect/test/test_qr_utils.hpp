// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test {

static inline
void check_qr(const string& root, const string& name_current_image, const string& config_name,
                const std::vector<Point>& corners,
                const std::vector<string>& decoded_info, const int max_pixel_error,
                bool isMulti = false) {
    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    FileNode images_list = file_config[config_name];
    size_t images_count = static_cast<size_t>(images_list.size());
    ASSERT_GT(images_count, 0u) << "Can't find validation data entries in 'test_images': " << dataset_config;
    for (size_t index = 0; index < images_count; index++) {
        FileNode config = images_list[(int)index];
        std::string name_test_image = config["image_name"];
        if (name_test_image == name_current_image) {
            if (isMulti) {
                for(int j = 0; j < int(corners.size()); j += 4) {
                    bool ok = false;
                    for (int k = 0; k < int(corners.size() / 4); k++) {
                        int count_eq_points = 0;
                        for (int i = 0; i < 4; i++) {
                            int x = config["x"][k][i];
                            int y = config["y"][k][i];
                            if(((abs(corners[j + i].x - x)) <= max_pixel_error) && ((abs(corners[j + i].y - y)) <= max_pixel_error))
                              count_eq_points++;
                        }
                        if (count_eq_points == 4) {
                            ok = true;
                            break;
                        }
                    }
                    EXPECT_TRUE(ok);
                }
            }
            else {
                for (int i = 0; i < (int)corners.size(); i++) {
                    int x = config["x"][i];
                    int y = config["y"][i];
                    EXPECT_NEAR(x, corners[i].x, max_pixel_error);
                    EXPECT_NEAR(y, corners[i].y, max_pixel_error);
                }
            }
#ifdef HAVE_QUIRC
            if (decoded_info.size() == 0ull)
                return;
            if (isMulti) {
                size_t count_eq_info = 0;
                for(int i = 0; i < int(decoded_info.size()); i++) {
                    for(int j = 0; j < int(decoded_info.size()); j++) {
                        std::string original_info = config["info"][j];
                        if(original_info == decoded_info[i]) {
                           count_eq_info++;
                           break;
                        }
                    }
                }
                EXPECT_EQ(decoded_info.size(), count_eq_info);
            }
            else {
                std::string original_info = config["info"];
                EXPECT_EQ(decoded_info[0], original_info);
            }
#endif
            return; // done
        }
    }
    FAIL() << "Not found results for '" << name_current_image << "' image in config file:" << dataset_config <<
              "Re-run tests with enabled UPDATE_QRCODE_TEST_DATA macro to update test data.\n";
}

}
