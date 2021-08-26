import os
import sys
import importlib

__all__ = ['init']

DEBUG = False
if hasattr(sys, 'OpenCV_LOADER_DEBUG'):
    DEBUG = True


def _load_extra_py_code_for_module(base, name):
    module_name = "{}.{}".format(__name__, name)
    export_module_name = "{}.{}".format(base, name)
    try:
        m = importlib.import_module(module_name)
    except ImportError:
        # Extension doesn't contain extra py code
        return False

    if DEBUG:
        print('OpenCV loader: added python code extension for: ' + name)

    if hasattr(m, '__all__'):
        export_members = {k: getattr(m, k) for k in m.__all__}
    else:
        export_members = m.__dict__

    # If it is C extension module it is already loaded by cv2 package
    if export_module_name in sys.modules:
        for k, v in export_members.items():
            if k.startswith('_'):  # skip internals
                continue
            if isinstance(v, type(sys)):  # don't bring modules
                continue
            if DEBUG: print('    symbol: {} = {}'.format(k, v))
            setattr(sys.modules[export_module_name], k, v)
    else:
        # Otherwise we should add this module to modules list manually
        if not hasattr(base, name):
            setattr(sys.modules[base], name, m)
        sys.modules[export_module_name] = m

    del sys.modules[module_name]
    return True


def _collect_extra_submodules():
    def modules_filter(module):
        # module is not internal
        return not module.startswith("_")

    __INIT_FILE_PATH = os.path.abspath(__file__)
    _extra_submodules_init_path = os.path.dirname(__INIT_FILE_PATH)
    return filter(modules_filter, os.listdir(_extra_submodules_init_path))


def init(base):
    _load_extra_py_code_for_module(base, '.cv2')  # special case

    for submodule in _collect_extra_submodules():
        if _load_extra_py_code_for_module(base, submodule):
            if DEBUG:
                print("Extra Python code for", submodule, "is loaded")
    del sys.modules[__name__]
