import sys
import os
import cv2 as cv


def add_argument(zoo, parser, name, help, required=False, default=None, type=None, action=None, nargs=None):
    if len(sys.argv) <= 1:
        return

    modelName = sys.argv[1]

    if os.path.isfile(zoo):
        fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
        node = fs.getNode(modelName)
        if not node.empty():
            value = node.getNode(name)
            if not value.empty():
                if value.isReal():
                    default = value.real()
                elif value.isString():
                    default = value.string()
                elif value.isInt():
                    default = int(value.real())
                elif value.isSeq():
                    default = []
                    for i in range(value.size()):
                        v = value.at(i)
                        if v.isInt():
                            default.append(int(v.real()))
                        elif v.isReal():
                            default.append(v.real())
                        else:
                            print('Unexpected value format')
                            exit(0)
                else:
                    print('Unexpected field format')
                    exit(0)
                required = False

    if action == 'store_true':
        default = 1 if default == 'true' else (0 if default == 'false' else default)
        assert(default is None or default == 0 or default == 1)
        parser.add_argument('--' + name, required=required, help=help, default=bool(default),
                            action=action)
    else:
        parser.add_argument('--' + name, required=required, help=help, default=default,
                            action=action, nargs=nargs, type=type)


def add_preproc_args(zoo, parser, sample):
    aliases = []
    if os.path.isfile(zoo):
        fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
        root = fs.root()
        for name in root.keys():
            model = root.getNode(name)
            if model.getNode('sample').string() == sample:
                aliases.append(name)

    parser.add_argument('alias', nargs='?', choices=aliases,
                        help='An alias name of model to extract preprocessing parameters from models.yml file.')
    add_argument(zoo, parser, 'model', required=True,
                 help='Path to a binary file of model contains trained weights. '
                      'It could be a file with extensions .caffemodel (Caffe), '
                      '.pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO)')
    add_argument(zoo, parser, 'config',
                 help='Path to a text file of model contains network configuration. '
                      'It could be a file with extensions .prototxt (Caffe), .pbtxt or .config (TensorFlow), .cfg (Darknet), .xml (OpenVINO)')
    add_argument(zoo, parser, 'mean', nargs='+', type=float, default=[0, 0, 0],
                 help='Preprocess input image by subtracting mean values. '
                      'Mean values should be in BGR order.')
    add_argument(zoo, parser, 'scale', type=float, default=1.0,
                 help='Preprocess input image by multiplying on a scale factor.')
    add_argument(zoo, parser, 'width', type=int,
                 help='Preprocess input image by resizing to a specific width.')
    add_argument(zoo, parser, 'height', type=int,
                 help='Preprocess input image by resizing to a specific height.')
    add_argument(zoo, parser, 'rgb', action='store_true',
                 help='Indicate that model works with RGB input images instead BGR ones.')
    add_argument(zoo, parser, 'classes',
                 help='Optional path to a text file with names of classes to label detected objects.')
    add_argument(zoo, parser, 'postprocessing', type=str,
                 help='Post-processing kind depends on model topology.')
    add_argument(zoo, parser, 'background_label_id', type=int, default=-1,
                 help='An index of background class in predictions. If not negative, exclude such class from list of classes.')


def findFile(filename):
    if filename:
        if os.path.exists(filename):
            return filename

        fpath = cv.samples.findFile(filename, False)
        if fpath:
            return fpath

        samplesDataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..',
                                      'data',
                                      'dnn')
        if os.path.exists(os.path.join(samplesDataDir, filename)):
            return os.path.join(samplesDataDir, filename)

        for path in ['OPENCV_DNN_TEST_DATA_PATH', 'OPENCV_TEST_DATA_PATH']:
            try:
                extraPath = os.environ[path]
                absPath = os.path.join(extraPath, 'dnn', filename)
                if os.path.exists(absPath):
                    return absPath
            except KeyError:
                pass

        print('File ' + filename + ' not found! Please specify a path to '
              '/opencv_extra/testdata in OPENCV_DNN_TEST_DATA_PATH environment '
              'variable or pass a full path to model.')
        exit(0)
