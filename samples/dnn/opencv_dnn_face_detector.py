#!/usr/bin/env python
import os
import sys
if hasattr(sys, 'dont_write_bytecode'): sys.dont_write_bytecode = True  # Don't write cache .pyc files

try:
    import cv2 as cv
except ImportError:
    print("FATAL: Can't find OpenCV Python module. If you've built it from sources without installation, "
          'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')
    raise

data_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../samples/data'))

# OpenCV face detector model files
face_detector_dir = os.path.join(data_path, 'dnn_face_detector')
config_path = os.path.join(face_detector_dir, 'opencv_face_detector.pbtxt')
model_path = os.path.join(os.environ.get('OPENCV_DNN_MODEL_WEIGHTS', face_detector_dir), 'opencv_face_detector_uint8.pb')

if not os.path.exists(config_path):
    raise Exception("Can't find face detector DNN model configuration: " + config_path);

def download_model_weights_or_die():
    try:
        sys.path.append(data_path)
        import opencv_data_downloader
    except ImportError:
        raise ImportError('Can\'t find OpenCV downloader module (should be located in samples/data directory).')

    metalink_file = os.path.join(face_detector_dir, 'download/opencv_dnn_face_detector_uint8.meta4')
    if not opencv_data_downloader.Downloader().download_metalink(metalink_file, dst_path=face_detector_dir):
        raise Exception("Can't download UINT8 DNN face detector model weights. "
                        "Refer to alternative download methods in README.md file: " + face_detector_dir);
    assert os.path.exists(model_path), model_path
    print("Model weights data has been downloaded: " + model_path)

if not os.path.exists(model_path):
    download_model_weights_or_die()

# Call common code for DNN-based object detection with specified DNN model
import object_detection
object_detection.runDNNObjectDetection(model = model_path, config = config_path,
                                       mean=[104, 177, 123], width=300, height=300)
sys.exit(0)
