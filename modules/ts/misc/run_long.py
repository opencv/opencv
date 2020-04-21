#!/usr/bin/env python
from __future__ import print_function
import xml.etree.ElementTree as ET
from glob import glob
from pprint import PrettyPrinter as PP

LONG_TESTS_DEBUG_VALGRIND = [
    ('calib3d', 'Calib3d_InitUndistortRectifyMap.accuracy', 2017.22),
    ('dnn', 'Reproducibility*', 1000),  # large DNN models
    ('dnn', '*RCNN*', 1000),  # very large DNN models
    ('dnn', '*RFCN*', 1000),  # very large DNN models
    ('dnn', '*EAST*', 1000),  # very large DNN models
    ('dnn', '*VGG16*', 1000),  # very large DNN models
    ('dnn', '*ZFNet*', 1000),  # very large DNN models
    ('dnn', '*ResNet101_DUC_HDC*', 1000),  # very large DNN models
    ('dnn', '*LResNet100E_IR*', 1000),  # very large DNN models
    ('dnn', '*read_yolo_voc_stream*', 1000),  # very large DNN models
    ('dnn', '*eccv16*', 1000),  # very large DNN models
    ('dnn', '*OpenPose*', 1000),  # very large DNN models
    ('dnn', '*SSD/*', 1000),  # very large DNN models
    ('face', 'CV_Face_FacemarkLBF.test_workflow', 10000.0), # >40min on i7
    ('features2d', 'Features2d/DescriptorImage.no_crash/3', 1000),
    ('features2d', 'Features2d/DescriptorImage.no_crash/4', 1000),
    ('features2d', 'Features2d/DescriptorImage.no_crash/5', 1000),
    ('features2d', 'Features2d/DescriptorImage.no_crash/6', 1000),
    ('features2d', 'Features2d/DescriptorImage.no_crash/7', 1000),
    ('imgcodecs', 'Imgcodecs_Png.write_big', 1000),  # memory limit
    ('imgcodecs', 'Imgcodecs_Tiff.decode_tile16384x16384', 1000),  # memory limit
    ('ml', 'ML_RTrees.regression', 1423.47),
    ('optflow', 'DenseOpticalFlow_DeepFlow.ReferenceAccuracy', 1360.95),
    ('optflow', 'DenseOpticalFlow_DeepFlow_perf.perf/0', 1881.59),
    ('optflow', 'DenseOpticalFlow_DeepFlow_perf.perf/1', 5608.75),
    ('optflow', 'DenseOpticalFlow_GlobalPatchColliderDCT.ReferenceAccuracy', 5433.84),
    ('optflow', 'DenseOpticalFlow_GlobalPatchColliderWHT.ReferenceAccuracy', 5232.73),
    ('optflow', 'DenseOpticalFlow_SimpleFlow.ReferenceAccuracy', 1542.1),
    ('photo', 'Photo_Denoising.speed', 1484.87),
    ('photo', 'Photo_DenoisingColoredMulti.regression', 2447.11),
    ('rgbd', 'Rgbd_Normals.compute', 1156.32),
    ('shape', 'Hauss.regression', 2625.72),
    ('shape', 'ShapeEMD_SCD.regression', 61913.7),
    ('shape', 'Shape_SCD.regression', 3311.46),
    ('tracking', 'AUKF.br_mean_squared_error', 10764.6),
    ('tracking', 'UKF.br_mean_squared_error', 5228.27),
    ('tracking', '*DistanceAndOverlap*/1', 1000.0), # dudek
    ('tracking', '*DistanceAndOverlap*/2', 1000.0), # faceocc2
    ('videoio', 'Videoio_Video.ffmpeg_writebig', 1000),
    ('xfeatures2d', 'Features2d_RotationInvariance_Descriptor_BoostDesc_LBGM.regression', 1124.51),
    ('xfeatures2d', 'Features2d_RotationInvariance_Descriptor_VGG120.regression', 2198.1),
    ('xfeatures2d', 'Features2d_RotationInvariance_Descriptor_VGG48.regression', 1958.52),
    ('xfeatures2d', 'Features2d_RotationInvariance_Descriptor_VGG64.regression', 2113.12),
    ('xfeatures2d', 'Features2d_RotationInvariance_Descriptor_VGG80.regression', 2167.16),
    ('xfeatures2d', 'Features2d_ScaleInvariance_Descriptor_BoostDesc_LBGM.regression', 1511.39),
    ('xfeatures2d', 'Features2d_ScaleInvariance_Descriptor_VGG120.regression', 1222.07),
    ('xfeatures2d', 'Features2d_ScaleInvariance_Descriptor_VGG48.regression', 1059.14),
    ('xfeatures2d', 'Features2d_ScaleInvariance_Descriptor_VGG64.regression', 1163.41),
    ('xfeatures2d', 'Features2d_ScaleInvariance_Descriptor_VGG80.regression', 1179.06),
    ('ximgproc', 'L0SmoothTest.SplatSurfaceAccuracy', 6382.26),
    ('ximgproc', 'perf*/1*:perf*/2*:perf*/3*:perf*/4*:perf*/5*:perf*/6*:perf*/7*:perf*/8*:perf*/9*', 1000.0),  # only first 10 parameters
    ('ximgproc', 'TypicalSet1/RollingGuidanceFilterTest.MultiThreadReproducibility/5', 1086.33),
    ('ximgproc', 'TypicalSet1/RollingGuidanceFilterTest.MultiThreadReproducibility/7', 1405.05),
    ('ximgproc', 'TypicalSet1/RollingGuidanceFilterTest.SplatSurfaceAccuracy/5', 1253.07),
    ('ximgproc', 'TypicalSet1/RollingGuidanceFilterTest.SplatSurfaceAccuracy/7', 1599.98),
    ('ximgproc', '*MultiThreadReproducibility*/1:*MultiThreadReproducibility*/2:*MultiThreadReproducibility*/3:*MultiThreadReproducibility*/4:*MultiThreadReproducibility*/5:*MultiThreadReproducibility*/6:*MultiThreadReproducibility*/7:*MultiThreadReproducibility*/8:*MultiThreadReproducibility*/9:*MultiThreadReproducibility*/1*', 1000.0),
    ('ximgproc', '*AdaptiveManifoldRefImplTest*/1:*AdaptiveManifoldRefImplTest*/2:*AdaptiveManifoldRefImplTest*/3', 1000.0),
    ('ximgproc', '*JointBilateralFilterTest_NaiveRef*', 1000.0),
    ('ximgproc', '*RollingGuidanceFilterTest_BilateralRef*/1*:*RollingGuidanceFilterTest_BilateralRef*/2*:*RollingGuidanceFilterTest_BilateralRef*/3*', 1000.0),
    ('ximgproc', '*JointBilateralFilterTest_NaiveRef*', 1000.0),
]


def longTestFilter(data, module=None):
    res = ['*', '-'] + [v for m, v, _time in data if module is None or m == module]
    return '--gtest_filter={}'.format(':'.join(res))


# Parse one xml file, filter out tests which took less than 'timeLimit' seconds
# Returns tuple: ( <module_name>, [ (<module_name>, <test_name>, <test_time>), ... ] )
def parseOneFile(filename, timeLimit):
    tree = ET.parse(filename)
    root = tree.getroot()

    def guess(s, delims):
        for delim in delims:
            tmp = s.partition(delim)
            if len(tmp[1]) != 0:
                return tmp[0]
        return None
    module = guess(filename, ['_posix_', '_nt_', '__']) or root.get('cv_module_name')
    if not module:
        return (None, None)
    res = []
    for elem in root.findall('.//testcase'):
        key = '{}.{}'.format(elem.get('classname'), elem.get('name'))
        val = elem.get('time')
        if float(val) >= timeLimit:
            res.append((module, key, float(val)))
    return (module, res)


# Parse all xml files in current folder and combine results into one list
# Print result to the stdout
if __name__ == '__main__':
    LIMIT = 1000
    res = []
    xmls = glob('*.xml')
    for xml in xmls:
        print('Parsing file', xml, '...')
        module, testinfo = parseOneFile(xml, LIMIT)
        if not module:
            print('SKIP')
            continue
        res.extend(testinfo)

    print('========= RESULTS =========')
    PP(indent=4, width=100).pprint(sorted(res))
