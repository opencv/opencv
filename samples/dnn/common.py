import sys
import os
import cv2 as cv


def add_argument(zoo, parser, name, help, required=False, default=None, type=None, action=None, nargs=None, alias=None):
    if alias is not None:
        modelName = alias
    elif len(sys.argv) > 1:
        modelName = sys.argv[1]
    else:
        return

    if os.path.isfile(zoo):
        fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
        node = fs.getNode(modelName)
        if not node.empty():
            value = node.getNode(name)
            if "sha1" in name:
                prefix = name.replace("sha1", "")
                value = node.getNode(prefix + "load_info")
                if prefix == "config_":
                    value = value.getNode("sha1")
                else:
                    value = value.getNode(name)
            if "download_sha" in name:
                prefix = name.replace("download_sha", "")
                value = node.getNode(prefix + "load_info")
                value = value.getNode(name)
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


def add_preproc_args(zoo, parser, sample, alias=None, prefix=""):
    aliases = []
    if os.path.isfile(zoo) and prefix == "":
        fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
        root = fs.root()
        for name in root.keys():
            model = root.getNode(name)
            if model.getNode('sample').string() == sample:
                aliases.append(name)
    if len(aliases):
        parser.add_argument(prefix+'alias', nargs='?', choices=aliases,
                            help='An alias name of model to extract preprocessing parameters from models.yml file.')

    add_argument(zoo, parser, prefix+'model',
                 help='Path to a binary file of model contains trained weights. '
                      'It could be a file with extensions .onnx (ONNX), '
                      '.tflite (TFLite), .bin (OpenVINO)', alias=alias)
    add_argument(zoo, parser, prefix+'config',
                 help='Path to a text file of model contains network configuration. '
                      'It could be a file with extension .xml (OpenVINO)', alias=alias)
    add_argument(zoo, parser, prefix+'mean', nargs='+', type=float, default=[0, 0, 0],
                 help='Preprocess input image by subtracting mean values. '
                      'Mean values should be in BGR order.', alias=alias)
    add_argument(zoo, parser, prefix+'std', nargs='+', type=float, default=[0, 0, 0],
                 help='Preprocess input image by dividing on a standard deviation.', alias=alias)
    add_argument(zoo, parser, prefix+'scale', type=float, default=1.0,
                 help='Preprocess input image by multiplying on a scale factor.', alias=alias)
    add_argument(zoo, parser, prefix+'width', type=int,
                 help='Preprocess input image by resizing to a specific width.', alias=alias)
    add_argument(zoo, parser, prefix+'height', type=int,
                 help='Preprocess input image by resizing to a specific height.', alias=alias)
    add_argument(zoo, parser, prefix+'rgb', action='store_true',
                 help='Indicate that model works with RGB input images instead BGR ones.', alias=alias)
    add_argument(zoo, parser, prefix+'labels',
                 help='Optional path to a text file with names of labels to label detected objects.', alias=alias)
    add_argument(zoo, parser, prefix+'postprocessing', type=str,
                 help='Post-processing kind depends on model topology.', alias=alias)
    add_argument(zoo, parser, prefix+'background_label_id', type=int, default=-1,
                 help='An index of background class in predictions. If not negative, exclude such class from list of classes.', alias=alias)
    add_argument(zoo, parser, prefix+'sha1', type=str,
                 help='Optional path to hashsum of downloaded model to be loaded from models.yml', alias=alias)
    add_argument(zoo, parser, prefix+'config_sha1', type=str,
                 help='Optional path to hashsum of downloaded config to be loaded from models.yml', alias=alias)
    add_argument(zoo, parser, prefix+'download_sha', type=str,
                 help='Optional path to hashsum of downloaded model to be loaded from models.yml', alias=alias)

def findModel(filename, sha1):
    if filename:
        if os.path.exists(filename):
            return filename

        fpath = cv.samples.findFile(filename, False)
        if fpath:
            return fpath

        if os.getenv('OPENCV_DOWNLOAD_CACHE_DIR') is None:
            print('[WARN] Please specify a path to model download directory in OPENCV_DOWNLOAD_CACHE_DIR environment variable.')
            return findFile(filename)

        if os.path.exists(os.path.join(os.environ['OPENCV_DOWNLOAD_CACHE_DIR'], sha1, filename)):
            return os.path.join(os.environ['OPENCV_DOWNLOAD_CACHE_DIR'], sha1, filename)

        if os.path.exists(os.path.join(os.environ['OPENCV_DOWNLOAD_CACHE_DIR'], filename)):
            return os.path.join(os.environ['OPENCV_DOWNLOAD_CACHE_DIR'], filename)

    raise FileNotFoundError('File ' + filename + ' not found! Please specify a path to '
            'model download directory in OPENCV_DOWNLOAD_CACHE_DIR '
            'environment variable or pass a full path to ' + filename)

def findFile(filename):
    if filename:
        if os.path.exists(filename):
            return filename

        fpath = cv.samples.findFile(filename, False)
        if fpath:
            return fpath

        if os.getenv('OPENCV_SAMPLES_DATA_PATH') is None:
            print('[WARN] Please specify a path to `/samples/data` in OPENCV_SAMPLES_DATA_PATH environment variable.')
            exit(0)

        if os.path.exists(os.path.join(os.environ['OPENCV_SAMPLES_DATA_PATH'], filename)):
            return os.path.join(os.environ['OPENCV_SAMPLES_DATA_PATH'], filename)

        for path in ['OPENCV_DNN_TEST_DATA_PATH', 'OPENCV_TEST_DATA_PATH', 'OPENCV_SAMPLES_DATA_PATH']:
            try:
                extraPath = os.environ[path]
                absPath = os.path.join(extraPath, 'dnn', filename)
                if os.path.exists(absPath):
                    return absPath
            except KeyError:
                pass

    raise FileNotFoundError(
        'File ' + filename + ' not found! Please specify the path to '
        '/opencv/samples/data in the OPENCV_SAMPLES_DATA_PATH environment variable, '
        'or specify the path to opencv_extra/testdata in the OPENCV_DNN_TEST_DATA_PATH environment variable, '
        'or specify the path to the model download cache directory in the OPENCV_DOWNLOAD_CACHE_DIR environment variable, '
        'or pass the full path to ' + filename + '.'
    )


def get_backend_id(backend_name):
    backend_ids = {
        "default": cv.dnn.DNN_BACKEND_DEFAULT,
        "openvino": cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
        "opencv": cv.dnn.DNN_BACKEND_OPENCV,
        "vkcom": cv.dnn.DNN_BACKEND_VKCOM,
        "cuda": cv.dnn.DNN_BACKEND_CUDA
    }

    if backend_name not in backend_ids:
        raise ValueError(f"Invalid backend name: {backend_name}")

    return backend_ids[backend_name]

def get_target_id(target_name):
    target_ids = {
        "cpu": cv.dnn.DNN_TARGET_CPU,
        "opencl": cv.dnn.DNN_TARGET_OPENCL,
        "opencl_fp16": cv.dnn.DNN_TARGET_OPENCL_FP16,
        "ncs2_vpu": cv.dnn.DNN_TARGET_MYRIAD,
        "hddl_vpu": cv.dnn.DNN_TARGET_HDDL,
        "vulkan": cv.dnn.DNN_TARGET_VULKAN,
        "cuda": cv.dnn.DNN_TARGET_CUDA,
        "cuda_fp16": cv.dnn.DNN_TARGET_CUDA_FP16
    }
    if target_name not in target_ids:
        raise ValueError(f"Invalid target name: {target_name}")

    return target_ids[target_name]

def get_backend_name(backend_id):
    backend_names = {
        cv.dnn.DNN_BACKEND_DEFAULT: "default",
        cv.dnn.DNN_BACKEND_INFERENCE_ENGINE: "openvino",
        cv.dnn.DNN_BACKEND_OPENCV: "opencv",
        cv.dnn.DNN_BACKEND_VKCOM: "vkcom",
        cv.dnn.DNN_BACKEND_CUDA: "cuda"
    }
    if hasattr(cv.dnn, 'DNN_BACKEND_INFERENCE_ENGINE_NGRAPH'):
        backend_names[cv.dnn.DNN_BACKEND_INFERENCE_ENGINE_NGRAPH] = "openvino"
    if hasattr(cv.dnn, 'DNN_BACKEND_TIMVX'):
        backend_names[cv.dnn.DNN_BACKEND_TIMVX] = "timvx"
    if hasattr(cv.dnn, 'DNN_BACKEND_CANN'):
        backend_names[cv.dnn.DNN_BACKEND_CANN] = "cann"
    if hasattr(cv.dnn, 'DNN_BACKEND_WEBNN'):
        backend_names[cv.dnn.DNN_BACKEND_WEBNN] = "webnn"
    return backend_names.get(backend_id, str(backend_id))


def get_engine_name(engine_id):
    engine_names = {
        getattr(cv.dnn, 'ENGINE_ORT', 2): "ort",
        getattr(cv.dnn, 'ENGINE_OPENCV', 1): "opencv",
        getattr(cv.dnn, 'ENGINE_AUTO', 0): "opencv"
    }
    return engine_names.get(engine_id, "opencv")


def get_target_name(target_id):
    target_names = {
        cv.dnn.DNN_TARGET_CPU: "cpu",
        cv.dnn.DNN_TARGET_OPENCL: "opencl",
        cv.dnn.DNN_TARGET_OPENCL_FP16: "opencl_fp16",
        cv.dnn.DNN_TARGET_MYRIAD: "ncs2_vpu",
        cv.dnn.DNN_TARGET_HDDL: "hddl_vpu",
        cv.dnn.DNN_TARGET_VULKAN: "vulkan",
        cv.dnn.DNN_TARGET_CUDA: "cuda",
        cv.dnn.DNN_TARGET_CUDA_FP16: "cuda_fp16"
    }
    return target_names.get(target_id, str(target_id))


def get_engine_id(engine_name):
    engine_ids = {
        "auto": getattr(cv.dnn, 'ENGINE_AUTO', 0),
        "opencv": getattr(cv.dnn, 'ENGINE_OPENCV', 1),
        "ort": getattr(cv.dnn, 'ENGINE_ORT', 2)
    }
    if engine_name not in engine_ids:
        raise ValueError(f"Invalid engine name: {engine_name}")

    return engine_ids[engine_name]


def resolve_engine_name(engine_name):
    if isinstance(engine_name, int):
        engine_map = {
            getattr(cv.dnn, 'ENGINE_AUTO', 0): "auto",
            getattr(cv.dnn, 'ENGINE_OPENCV', 1): "opencv",
            getattr(cv.dnn, 'ENGINE_ORT', 2): "ort"
        }
        engine_name = engine_map.get(engine_name, "auto")

    forced = os.environ.get("OPENCV_FORCE_DNN_ENGINE")
    if forced is not None:
        try:
            forced_val = int(forced)
            if forced_val == getattr(cv.dnn, 'ENGINE_OPENCV', 1):
                engine_name = "opencv"
            elif forced_val == getattr(cv.dnn, 'ENGINE_ORT', 2):
                engine_name = "ort"
        except ValueError:
            pass

    if engine_name == "auto" or not engine_name:
        engine_name = "opencv"

    if engine_name == "ort":
        ort_available = False
        try:
            build_info = cv.getBuildInformation()
            for line in build_info.splitlines():
                if "ONNX Runtime:" in line and "YES" in line:
                    ort_available = True
                    break
        except Exception:
            pass
        if not ort_available:
            engine_name = "opencv"

    return engine_name


def resolve_backend_name(backend_name):
    if isinstance(backend_name, int):
        if backend_name == getattr(cv.dnn, 'DNN_BACKEND_DEFAULT', 0):
            forced = os.environ.get("OPENCV_DNN_BACKEND_DEFAULT")
            if forced is not None:
                try:
                    return get_backend_name(int(forced))
                except ValueError:
                    pass
            return "opencv"
        return get_backend_name(backend_name)
    elif isinstance(backend_name, str):
        if backend_name == "default":
            forced = os.environ.get("OPENCV_DNN_BACKEND_DEFAULT")
            if forced is not None:
                try:
                    return get_backend_name(int(forced))
                except ValueError:
                    pass
            return "opencv"
        return backend_name
    return str(backend_name)


def print_dnn_config(net_or_engine, backend_name=None, target_name=None):
    if backend_name is None and target_name is None and hasattr(net_or_engine, 'getEngine'):
        net = net_or_engine
        engine = get_engine_name(net.getEngine())
        backend = get_backend_name(net.getPreferableBackend())
        target = get_target_name(net.getPreferableTarget())
        print(f"DNN Engine: {engine}, Backend: {backend}, Target: {target}")
        return

    engine_resolved = resolve_engine_name(net_or_engine)
    backend_resolved = resolve_backend_name(backend_name)
    if isinstance(target_name, int):
        target_name = get_target_name(target_name)
    print(f"DNN Engine: {engine_resolved}, Backend: {backend_resolved}, Target: {target_name}")
