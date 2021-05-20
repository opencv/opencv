import sys
import importlib

__all__ = ['init']


DEBUG = False
if hasattr(sys, 'OpenCV_LOADER_DEBUG'):
    DEBUG = True


def _load_py_code(base, name):
    try:
        m = importlib.import_module(__name__ + name)
    except ImportError:
        return  # extension doesn't exist?

    if DEBUG: print('OpenCV loader: added python code extension for: ' + name)

    if hasattr(m, '__all__'):
        export_members = { k : getattr(m, k) for k in m.__all__ }
    else:
        export_members = m.__dict__

    for k, v in export_members.items():
        if k.startswith('_'):  # skip internals
            continue
        if isinstance(v, type(sys)):  # don't bring modules
            continue
        if DEBUG: print('    symbol: {} = {}'.format(k, v))
        setattr(sys.modules[base + name ], k, v)

    del sys.modules[__name__ + name]


# TODO: listdir
def init(base):
    _load_py_code(base, '.cv2')  # special case
    prefix = base
    prefix_len = len(prefix)

    modules = [ m for m in sys.modules.keys() if m.startswith(prefix) ]
    for m in modules:
        m2 = m[prefix_len:]  # strip prefix
        if len(m2) == 0:
            continue
        if m2.startswith('._'):  # skip internals
            continue
        if m2.startswith('.load_config_'):  # skip helper files
            continue
        _load_py_code(base, m2)

    del sys.modules[__name__]
