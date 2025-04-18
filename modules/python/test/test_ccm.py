#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2 as cv
import tempfile

from tests_common import NewOpenCVTests

class photo_test(NewOpenCVTests):

    def setUp(self):
        super(photo_test, self).setUp()
        self.image_cache = {}

    def test_model(self):
        s = np.array([
            [214.11, 98.67, 37.97],
            [231.94, 153.1, 85.27],
            [204.08, 143.71, 78.46],
            [190.58, 122.99, 30.84],
            [230.93, 148.46, 100.84],
            [228.64, 206.97, 97.5],
            [229.09, 137.07, 55.29],
            [189.21, 111.22, 92.66],
            [223.5, 96.42, 75.45],
            [201.82, 69.71, 50.9],
            [240.52, 196.47, 59.3],
            [235.73, 172.13, 54.],
            [131.6, 75.04, 68.86],
            [189.04, 170.43, 42.05],
            [222.23, 74., 71.95],
            [241.01, 199.1, 61.15],
            [224.99, 101.4, 100.24],
            [174.58, 152.63, 91.52],
            [248.06, 227.69, 140.5],
            [241.15, 201.38, 115.58],
            [236.49, 175.87, 88.86],
            [212.19, 133.49, 54.79],
            [181.17, 102.94, 36.18],
            [115.1, 53.77, 15.23]
        ], dtype=np.float64)

        src = (s / 255.).astype(np.float64).reshape(-1, 1, 3)
        model = cv.ccm.ColorCorrectionModel(src, cv.ccm.COLORCHECKER_MACBETH)
        colorCorrectionMat = model.compute()

        src_rgbl = np.array([
            [0.68078957, 0.12382801, 0.01514889],
            [0.81177942, 0.32550452, 0.089818],
            [0.61259378, 0.2831933, 0.07478902],
            [0.52696493, 0.20105976, 0.00958657],
            [0.80402284, 0.30419523, 0.12989841],
            [0.78658646, 0.63184111, 0.12062068],
            [0.78999637, 0.25520249, 0.03462853],
            [0.51866697, 0.16114393, 0.1078387],
            [0.74820768, 0.11770076, 0.06862177],
            [0.59776825, 0.05765816, 0.02886627],
            [0.8793145, 0.56346033, 0.0403954],
            [0.84124847, 0.42120746, 0.03287592],
            [0.23333214, 0.06780408, 0.05612276],
            [0.5176423, 0.41210976, 0.01896255],
            [0.73888613, 0.06575388, 0.06181293],
            [0.88326036, 0.58018751, 0.04321991],
            [0.75922531, 0.13149072, 0.1282041],
            [0.4345097, 0.32331019, 0.10494139],
            [0.94110142, 0.77941419, 0.26946323],
            [0.88438952, 0.5949049, 0.17536928],
            [0.84722687, 0.44160449, 0.09834799],
            [0.66743106, 0.24076803, 0.03394333],
            [0.47141286, 0.13592419, 0.01362205],
            [0.17377101, 0.03256864, 0.00203026]
        ], dtype=np.float64)
        np.testing.assert_allclose(src_rgbl, model.getSrcLinearRGB().reshape(-1, 3), rtol=1e-4, atol=1e-4)

        dst_rgbl = np.array([
            [0.17303173, 0.08211037, 0.05672686],
            [0.56832031, 0.29269488, 0.21835529],
            [0.10365019, 0.19588357, 0.33140475],
            [0.10159676, 0.14892193, 0.05188294],
            [0.22159627, 0.21584476, 0.43461196],
            [0.10806379, 0.51437196, 0.41264213],
            [0.74736423, 0.20062878, 0.02807988],
            [0.05757947, 0.10516793, 0.40296109],
            [0.56676218, 0.08424805, 0.11969461],
            [0.11099515, 0.04230796, 0.14292554],
            [0.34546869, 0.50872001, 0.04944204],
            [0.79461323, 0.35942459, 0.02051968],
            [0.01710416, 0.05022043, 0.29220674],
            [0.05598012, 0.30021149, 0.06871162],
            [0.45585457, 0.03033727, 0.04085654],
            [0.85737614, 0.56757335, 0.0068503],
            [0.53348585, 0.08861148, 0.30750446],
            [-0.0374061, 0.24699498, 0.40041217],
            [0.91262695, 0.91493909, 0.89367049],
            [0.57981916, 0.59200418, 0.59328881],
            [0.35490581, 0.36544831, 0.36755375],
            [0.19007357, 0.19186587, 0.19308397],
            [0.08529188, 0.08887994, 0.09257601],
            [0.0303193, 0.03113818, 0.03274845]
        ], dtype=np.float64)
        np.testing.assert_allclose(dst_rgbl, model.getRefLinearRGB().reshape(-1, 3), rtol=1e-4, atol=1e-4)

        mask = np.ones((24, 1), dtype=np.uint8)
        np.testing.assert_allclose(model.getMask(), mask, rtol=0.0, atol=0.0)

        # Test reference color matrix
        refColorMat = np.array([
            [0.37406520, 0.02066507, 0.05804047],
            [0.12719672, 0.77389268, -0.01569404],
            [-0.27627010, 0.00603427, 2.74272981]
        ], dtype=np.float64)
        np.testing.assert_allclose(colorCorrectionMat, refColorMat, rtol=1e-4, atol=1e-4)

    def test_masks_weights_1(self):
        s = np.array([
            [214.11, 98.67, 37.97],
            [231.94, 153.1, 85.27],
            [204.08, 143.71, 78.46],
            [190.58, 122.99, 30.84],
            [230.93, 148.46, 100.84],
            [228.64, 206.97, 97.5],
            [229.09, 137.07, 55.29],
            [189.21, 111.22, 92.66],
            [223.5, 96.42, 75.45],
            [201.82, 69.71, 50.9],
            [240.52, 196.47, 59.3],
            [235.73, 172.13, 54.],
            [131.6, 75.04, 68.86],
            [189.04, 170.43, 42.05],
            [222.23, 74., 71.95],
            [241.01, 199.1, 61.15],
            [224.99, 101.4, 100.24],
            [174.58, 152.63, 91.52],
            [248.06, 227.69, 140.5],
            [241.15, 201.38, 115.58],
            [236.49, 175.87, 88.86],
            [212.19, 133.49, 54.79],
            [181.17, 102.94, 36.18],
            [115.1, 53.77, 15.23]
        ], dtype=np.float64)

        weightsList = np.array([1.1, 0, 0, 1.2, 0, 0, 1.3, 0, 0, 1.4, 0, 0,
                               0.5, 0, 0, 0.6, 0, 0, 0.7, 0, 0, 0.8, 0, 0], dtype=np.float64)
        weightsList = weightsList.reshape(-1, 1)

        src = (s / 255.).astype(np.float64).reshape(-1, 1, 3)
        model = cv.ccm.ColorCorrectionModel(src, cv.ccm.COLORCHECKER_MACBETH)
        model.setColorSpace(cv.ccm.COLOR_SPACE_SRGB)
        model.setCcmType(cv.ccm.CCM_LINEAR)
        model.setDistance(cv.ccm.DISTANCE_CIE2000)
        model.setLinearization(cv.ccm.LINEARIZATION_GAMMA)
        model.setLinearizationGamma(2.2)
        model.setLinearizationDegree(3)
        model.setSaturatedThreshold(0, 0.98)
        model.setWeightsList(weightsList)
        model.setWeightCoeff(1.5)
        _ = model.compute()

        weights = np.array([1.15789474, 1.26315789, 1.36842105, 1.47368421,
                           0.52631579, 0.63157895, 0.73684211, 0.84210526], dtype=np.float64)
        np.testing.assert_allclose(model.getWeights(), weights.reshape(-1, 1), rtol=1e-4, atol=1e-4)

        mask = np.array([True, False, False, True, False, False,
                        True, False, False, True, False, False,
                        True, False, False, True, False, False,
                        True, False, False, True, False, False], dtype=np.uint8)
        np.testing.assert_allclose(model.getMask(), mask.reshape(-1, 1), rtol=0.0, atol=0.0)

    def test_masks_weights_2(self):
        s = np.array([
            [214.11, 98.67, 37.97],
            [231.94, 153.1, 85.27],
            [204.08, 143.71, 78.46],
            [190.58, 122.99, 30.84],
            [230.93, 148.46, 100.84],
            [228.64, 206.97, 97.5],
            [229.09, 137.07, 55.29],
            [189.21, 111.22, 92.66],
            [223.5, 96.42, 75.45],
            [201.82, 69.71, 50.9],
            [240.52, 196.47, 59.3],
            [235.73, 172.13, 54.],
            [131.6, 75.04, 68.86],
            [189.04, 170.43, 42.05],
            [222.23, 74., 71.95],
            [241.01, 199.1, 61.15],
            [224.99, 101.4, 100.24],
            [174.58, 152.63, 91.52],
            [248.06, 227.69, 140.5],
            [241.15, 201.38, 115.58],
            [236.49, 175.87, 88.86],
            [212.19, 133.49, 54.79],
            [181.17, 102.94, 36.18],
            [115.1, 53.77, 15.23]
        ], dtype=np.float64)

        src = (s / 255.).astype(np.float64).reshape(-1, 1, 3)
        model = cv.ccm.ColorCorrectionModel(src, cv.ccm.COLORCHECKER_MACBETH)
        model.setCcmType(cv.ccm.CCM_LINEAR)
        model.setDistance(cv.ccm.DISTANCE_CIE2000)
        model.setLinearization(cv.ccm.LINEARIZATION_GAMMA)
        model.setLinearizationGamma(2.2)
        model.setLinearizationDegree(3)
        model.setSaturatedThreshold(0.05, 0.93)
        model.setWeightsList(np.array([]))
        model.setWeightCoeff(1.5)
        _ = model.compute()

        weights = np.array([
            0.65554256, 1.49454705, 1.00499244, 0.79735434, 1.16327759,
            1.68623868, 1.37973155, 0.73213388, 1.0169629, 0.47430246,
            1.70312161, 0.45414218, 1.15910007, 0.7540434, 1.05049802,
            1.04551645, 1.54082353, 1.02453421, 0.6015915, 0.26154558
        ], dtype=np.float64)
        np.testing.assert_allclose(model.getWeights(), weights.reshape(-1, 1), rtol=1e-4, atol=1e-4)

        # Test mask
        mask = np.array([True, True, True, True, True, True,
                        True, True, True, True, False, True,
                        True, True, True, False, True, True,
                        False, False, True, True, True, True], dtype=np.uint8)
        np.testing.assert_allclose(model.getMask(), mask.reshape(-1, 1), rtol=0.0, atol=0.0)

    def test_compute_color_correction_matrix(self):
        path = self.find_file('cv/mcc/mcc_ccm_test.yml')
        fs = cv.FileStorage(path, cv.FileStorage_READ)
        chartsRGB = fs.getNode("chartsRGB").mat()

        src = (chartsRGB[:, 1].reshape(-1, 1, 3) / 255.).astype(np.float64)

        model = cv.ccm.ColorCorrectionModel(src, cv.ccm.COLORCHECKER_MACBETH)
        colorCorrectionMat = model.compute()

        gold_ccm = fs.getNode("ccm").mat()
        fs.release()

        np.testing.assert_allclose(gold_ccm, colorCorrectionMat, rtol=1e-8, atol=1e-8)

        gold_loss = 4.6386569120323129
        loss = model.getLoss()
        self.assertAlmostEqual(gold_loss, loss, places=8)

    def test_correctImage(self):
        img = self.get_sample('cv/mcc/mcc_ccm_test.jpg')
        self.assertIsNotNone(img, "Test image can't be loaded: ")

        gold_img = self.get_sample('cv/mcc/mcc_ccm_test_res.png')
        self.assertIsNotNone(gold_img, "Ground truth for test image can't be loaded: ")

        path = self.find_file("cv/mcc/mcc_ccm_test.yml")
        fs = cv.FileStorage(path, cv.FileStorage_READ)
        chartsRGB = fs.getNode("chartsRGB").mat()
        fs.release()

        src = (chartsRGB[:, 1].reshape(-1, 1, 3) / 255.).astype(np.float64)
  
        np.savetxt('src_test_correct.txt',src.reshape(-1,3),fmt="%.2f")
        model = cv.ccm.ColorCorrectionModel(src, cv.ccm.COLORCHECKER_MACBETH)
        _ = model.compute()

        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = image.astype(np.float64) / 255.
        calibratedImage = np.zeros_like(image)
        model.correctImage(image, calibratedImage)
        calibratedImage = np.clip(np.rint(calibratedImage * 255), 0, 255).astype(np.uint8)
        calibratedImage = cv.cvtColor(calibratedImage, cv.COLOR_RGB2BGR)

        np.testing.assert_allclose(gold_img, calibratedImage, rtol=0.1, atol=0.1)

    def test_mcc_ccm_combined(self):
        detector = cv.mcc_CCheckerDetector.create()

        img = self.get_sample('cv/mcc/mcc_ccm_test.jpg')
        self.assertIsNotNone(img, "Test image can't be loaded: ")
    
        gold_img = self.get_sample('cv/mcc/mcc_ccm_test_res.png')
        self.assertIsNotNone(gold_img, "Ground truth for test image can't be loaded: ")

        detector.setColorChartType(cv.mcc.MCC24)
        self.assertTrue(detector.process(img))

        checkers = detector.getListColorChecker()
        # Get colors from detector and save for debugging
        src = checkers[0].getChartsRGB(False).reshape(-1, 1, 3) / 255.
        src = src.astype(np.float64)
        
        # Load reference colors from file for comparison
        path = self.find_file('cv/mcc/mcc_ccm_test.yml')
        fs = cv.FileStorage(path, cv.FileStorage_READ)
        chartsRGB = fs.getNode("chartsRGB").mat()
        ref_src = (chartsRGB[:, 1].reshape(-1, 1, 3) / 255.).astype(np.float64)
        fs.release()
        
        # Verify that detected colors are close to reference colors
        np.testing.assert_allclose(src, ref_src, rtol=0.01, atol=0.01)
        
        # Use reference colors for model computation
        model = cv.ccm.ColorCorrectionModel(ref_src, cv.ccm.COLORCHECKER_MACBETH)
        _ = model.compute()

        # Convert image to float64 and normalize
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = image.astype(np.float64) / 255.
        calibratedImage = np.zeros_like(image)
        model.correctImage(image, calibratedImage)
<<<<<<< HEAD
        
        # Ensure values are in valid range before conversion
        calibratedImage = np.clip(calibratedImage, 0.0, 1.0)
        
        # Convert to uint8 with explicit rounding
        calibratedImage = (calibratedImage * 255.0).astype(np.float64)
        calibratedImage = np.rint(calibratedImage).astype(np.uint8)
=======
        calibratedImage = np.clip(calibratedImage * 255.0 + 0.5, 0, 255).astype(np.uint8)
>>>>>>> 144bf9b09a96edc365d5142131686f725aa47021
        calibratedImage = cv.cvtColor(calibratedImage, cv.COLOR_RGB2BGR)
        
        np.testing.assert_allclose(gold_img, calibratedImage, rtol=0.1, atol=0.1)

    def test_serialization(self):
        path1 = self.find_file("cv/mcc/mcc_ccm_test.yml")
        fs = cv.FileStorage(path1, cv.FileStorage_READ)
        chartsRGB = fs.getNode("chartsRGB").mat()
        fs.release()

        model = cv.ccm.ColorCorrectionModel(chartsRGB[:, 1].reshape(-1,  1, 3) / 255., cv.ccm.COLORCHECKER_MACBETH)
        _ = model.compute()

        path1 = tempfile.mktemp(suffix='.yaml')
        fs1 = cv.FileStorage(path1, cv.FileStorage_WRITE)
        model.write(fs1)
        fs1.release()

        model1 = cv.ccm.ColorCorrectionModel()
        fs2 = cv.FileStorage(path1, cv.FileStorage_READ)
        modelNode = fs2.getNode("ColorCorrectionModel")
        model1.read(modelNode)
        fs2.release()

        path2 = tempfile.mktemp(suffix='.yaml')
        fs3 = cv.FileStorage(path2, cv.FileStorage_WRITE)
        model1.write(fs3)
        fs3.release()

        with open(path1, 'r') as file1:
            str1 = file1.read()
        with open(path2, 'r') as file2:
            str2 = file2.read()
        self.assertEqual(str1, str2)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()