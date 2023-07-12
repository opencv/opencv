'''
OpenCV Python binary extension loader
'''
import os
import importlib
import sys

__all__ = []

try:
    import numpy
    import numpy.core.multiarray
except ImportError:
    print('OpenCV bindings requires "numpy" package.')
    print('Install it via command:')
    print('    pip install numpy')
    raise

# TODO
# is_x64 = sys.maxsize > 2**32


def __load_extra_py_code_for_module(base, name, enable_debug_print=False):
    module_name = "{}.{}".format(__name__, name)
    export_module_name = "{}.{}".format(base, name)
    native_module = sys.modules.pop(module_name, None)
    try:
        py_module = importlib.import_module(module_name)
    except ImportError as err:
        if enable_debug_print:
            print("Can't load Python code for module:", module_name,
                  ". Reason:", err)
        # Extension doesn't contain extra py code
        return False

    if not hasattr(base, name):
        setattr(sys.modules[base], name, py_module)
    sys.modules[export_module_name] = py_module
    # If it is C extension module it is already loaded by cv2 package
    if native_module:
        setattr(py_module, "_native", native_module)
        for k, v in filter(lambda kv: not hasattr(py_module, kv[0]),
                           native_module.__dict__.items()):
            if enable_debug_print: print('    symbol({}): {} = {}'.format(name, k, v))
            setattr(py_module, k, v)
    return True


def __collect_extra_submodules(enable_debug_print=False):
    def modules_filter(module):
        return all((
             # module is not internal
             not module.startswith("_"),
             not module.startswith("python-"),
             # it is not a file
             os.path.isdir(os.path.join(_extra_submodules_init_path, module))
        ))
    if sys.version_info[0] < 3:
        if enable_debug_print:
            print("Extra submodules is loaded only for Python 3")
        return []

    __INIT_FILE_PATH = os.path.abspath(__file__)
    _extra_submodules_init_path = os.path.dirname(__INIT_FILE_PATH)
    return filter(modules_filter, os.listdir(_extra_submodules_init_path))


def bootstrap():
    import sys

    import copy
    save_sys_path = copy.copy(sys.path)

    if hasattr(sys, 'OpenCV_LOADER'):
        print(sys.path)
        raise ImportError('ERROR: recursion is detected during loading of "cv2" binary extensions. Check OpenCV installation.')
    sys.OpenCV_LOADER = True

    DEBUG = False
    if hasattr(sys, 'OpenCV_LOADER_DEBUG'):
        DEBUG = True

    import platform
    if DEBUG: print('OpenCV loader: os.name="{}"  platform.system()="{}"'.format(os.name, str(platform.system())))

    LOADER_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    PYTHON_EXTENSIONS_PATHS = []
    BINARIES_PATHS = []

    g_vars = globals()
    l_vars = locals().copy()

    if sys.version_info[:2] < (3, 0):
        from . load_config_py2 import exec_file_wrapper
    else:
        from . load_config_py3 import exec_file_wrapper

    def load_first_config(fnames, required=True):
        for fname in fnames:
            fpath = os.path.join(LOADER_DIR, fname)
            if not os.path.exists(fpath):
                if DEBUG: print('OpenCV loader: config not found, skip: {}'.format(fpath))
                continue
            if DEBUG: print('OpenCV loader: loading config: {}'.format(fpath))
            exec_file_wrapper(fpath, g_vars, l_vars)
            return True
        if required:
            raise ImportError('OpenCV loader: missing configuration file: {}. Check OpenCV installation.'.format(fnames))

    load_first_config(['config.py'], True)
    load_first_config([
        'config-{}.{}.py'.format(sys.version_info[0], sys.version_info[1]),
        'config-{}.py'.format(sys.version_info[0])
    ], True)

    if DEBUG: print('OpenCV loader: PYTHON_EXTENSIONS_PATHS={}'.format(str(l_vars['PYTHON_EXTENSIONS_PATHS'])))
    if DEBUG: print('OpenCV loader: BINARIES_PATHS={}'.format(str(l_vars['BINARIES_PATHS'])))

    applySysPathWorkaround = False
    if hasattr(sys, 'OpenCV_REPLACE_SYS_PATH_0'):
        applySysPathWorkaround = True
    else:
        try:
            BASE_DIR = os.path.dirname(LOADER_DIR)
            if sys.path[0] == BASE_DIR or os.path.realpath(sys.path[0]) == BASE_DIR:
                applySysPathWorkaround = True
        except:
            if DEBUG: print('OpenCV loader: exception during checking workaround for sys.path[0]')
            pass  # applySysPathWorkaround is False

    for p in reversed(l_vars['PYTHON_EXTENSIONS_PATHS']):
        sys.path.insert(1 if not applySysPathWorkaround else 0, p)

    if os.name == 'nt':
        if sys.version_info[:2] >= (3, 8):  # https://github.com/python/cpython/pull/12302
            for p in l_vars['BINARIES_PATHS']:
                try:
                    os.add_dll_directory(p)
                except Exception as e:
                    if DEBUG: print('Failed os.add_dll_directory(): '+ str(e))
                    pass
        os.environ['PATH'] = ';'.join(l_vars['BINARIES_PATHS']) + ';' + os.environ.get('PATH', '')
        if DEBUG: print('OpenCV loader: PATH={}'.format(str(os.environ['PATH'])))
    else:
        # amending of LD_LIBRARY_PATH works for sub-processes only
        os.environ['LD_LIBRARY_PATH'] = ':'.join(l_vars['BINARIES_PATHS']) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

    if DEBUG: print("Relink everything from native cv2 module to cv2 package")

    py_module = sys.modules.pop("cv2")

    native_module = importlib.import_module("cv2")

    sys.modules["cv2"] = py_module
    setattr(py_module, "_native", native_module)

    for item_name, item in filter(lambda kv: kv[0] not in ("__file__", "__loader__", "__spec__",
                                                           "__name__", "__package__"),
                                  native_module.__dict__.items()):
        if item_name not in g_vars:
            g_vars[item_name] = item

    sys.path = save_sys_path  # multiprocessing should start from bootstrap code (https://github.com/opencv/opencv/issues/18502)

    try:
        del sys.OpenCV_LOADER
    except Exception as e:
        if DEBUG:
            print("Exception during delete OpenCV_LOADER:", e)

    if DEBUG: print('OpenCV loader: binary extension... OK')

    for submodule in __collect_extra_submodules(DEBUG):
        if __load_extra_py_code_for_module("cv2", submodule, DEBUG):
            if DEBUG: print("Extra Python code for", submodule, "is loaded")

    if DEBUG: print('OpenCV loader: DONE')


bootstrap()
