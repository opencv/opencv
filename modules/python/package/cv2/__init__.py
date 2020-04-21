'''
OpenCV Python binary extension loader
'''
import os
import sys

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

def bootstrap():
    import sys
    if hasattr(sys, 'OpenCV_LOADER'):
        print(sys.path)
        raise ImportError('ERROR: recursion is detected during loading of "cv2" binary extensions. Check OpenCV installation.')
    sys.OpenCV_LOADER = True

    DEBUG = False
    if hasattr(sys, 'OpenCV_LOADER_DEBUG'):
        DEBUG = True

    import platform
    if DEBUG: print('OpenCV loader: os.name="{}"  platform.system()="{}"'.format(os.name, str(platform.system())))

    LOADER_DIR=os.path.dirname(os.path.abspath(__file__))

    PYTHON_EXTENSIONS_PATHS = []
    BINARIES_PATHS = []

    g_vars = globals()
    l_vars = locals()

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

    for p in reversed(l_vars['PYTHON_EXTENSIONS_PATHS']):
        sys.path.insert(1, p)

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

    if DEBUG: print('OpenCV loader: replacing cv2 module')
    del sys.modules['cv2']
    import cv2

    try:
        import sys
        del sys.OpenCV_LOADER
    except:
        pass

    if DEBUG: print('OpenCV loader: DONE')

bootstrap()
