from tests_common import NewOpenCVTests, unittest
import cv2 as cv
import os


def import_path():
    import sys
    if sys.version_info[0] < 3 or sys.version_info[1] < 6:
        raise unittest.SkipTest('Python 3.6+ required')

    from pathlib import Path
    return Path


class CanPassPathLike(NewOpenCVTests):
    def test_pathlib_path(self):
        Path = import_path()

        img_path = self.find_file('cv/imgproc/stuff.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])

        image_from_str = cv.imread(img_path)
        self.assertIsNotNone(image_from_str)

        image_from_path = cv.imread(Path(img_path))
        self.assertIsNotNone(image_from_path)


    def test_type_mismatch(self):
        import_path() # checks python version

        with self.assertRaises(cv.error) as context:
            cv.imread(123)

        self.assertTrue('str or path-like' in str(context.exception))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
