import gdb
import numpy as np
from enum import Enum

np.set_printoptions(suppress=True)  # prevent numpy exponential notation on print, default False
# np.set_printoptions(threshold=sys.maxsize)


def conv(obj, t):
    return gdb.parse_and_eval(f'({t})({obj})')


def booli(obj):
    return conv(str(obj).lower(), 'bool')


def stri(obj):
    s = f'"{obj}"'
    return conv(s.translate(s.maketrans('\n', ' ')), 'char*')


class MagicValues(Enum):
    MAGIC_VAL = 0x42FF0000
    AUTO_STEP = 0
    CONTINUOUS_FLAG = 1 << 14
    SUBMATRIX_FLAG = 1 << 15


class MagicMasks(Enum):
    MAGIC_MASK = 0xFFFF0000
    TYPE_MASK = 0x00000FFF
    DEPTH_MASK = 7


class Depth(Enum):
    CV_8U = 0
    CV_8S = 1
    CV_16U = 2
    CV_16S = 3
    CV_32S = 4
    CV_32F = 5
    CV_64F = 6
    CV_16F = 7


def create_enum(n):
    def make_type(depth, cn):
        return depth.value + ((cn - 1) << 3)
    defs = [(f'{depth.name}C{i}', make_type(depth, i)) for depth in Depth for i in range(1, n + 1)]
    return Enum('Type', defs)


Type = create_enum(512)


class Flags:
    def depth(self):
        return Depth(self.flags & MagicMasks.DEPTH_MASK.value)

    def dtype(self):
        depth = self.depth()
        ret = None

        if depth == Depth.CV_8U:
            ret = (np.uint8, 'uint8_t')
        elif depth == Depth.CV_8S:
            ret = (np.int8, 'int8_t')
        elif depth == Depth.CV_16U:
            ret = (np.uint16, 'uint16_t')
        elif depth == Depth.CV_16S:
            ret = (np.int16, 'int16_t')
        elif depth == Depth.CV_32S:
            ret = (np.int32, 'int32_t')
        elif depth == Depth.CV_32F:
            ret = (np.float32, 'float')
        elif depth == Depth.CV_64F:
            ret = (np.float64, 'double')
        elif depth == Depth.CV_16F:
            ret = (np.float16, 'float16')

        return ret

    def type(self):
        return Type(self.flags & MagicMasks.TYPE_MASK.value)

    def channels(self):
        return ((self.flags & (511 << 3)) >> 3) + 1

    def is_continuous(self):
        return (self.flags & MagicValues.CONTINUOUS_FLAG.value) != 0

    def is_submatrix(self):
        return (self.flags & MagicValues.SUBMATRIX_FLAG.value) != 0

    def __init__(self, flags):
        self.flags = flags

    def __iter__(self):
        return iter({
                        'type': stri(self.type().name),
                        'is_continuous': booli(self.is_continuous()),
                        'is_submatrix': booli(self.is_submatrix())
                    }.items())


class Size:
    def __init__(self, ptr):
        self.ptr = ptr

    def dims(self):
        return int((self.ptr - 1).dereference())

    def to_numpy(self):
        return np.array([int(self.ptr[i]) for i in range(self.dims())], dtype=np.int64)

    def __iter__(self):
        return iter({'size': stri(self.to_numpy())}.items())


class Mat:
    def __init__(self, m, size, flags):
        (dtype, ctype) = flags.dtype()
        elsize = np.dtype(dtype).itemsize

        shape = size.to_numpy()
        steps = np.asarray([int(m['step']['p'][i]) for i in range(len(shape))], dtype=np.int64)

        ptr = m['data']
        # either we are default-constructed or sizes are zero
        if int(ptr) == 0 or np.prod(shape * steps) == 0:
            self.mat = np.array([])
            self.view = self.mat
            return

        # we don't want to show excess brackets
        if flags.channels() != 1:
            shape = np.append(shape, flags.channels())
            steps = np.append(steps, elsize)

        # get the length of contiguous array from data to the last element of the matrix
        length = 1 + np.sum((shape - 1) * steps) // elsize

        if dtype != np.float16:
            # read all elements into self.mat
            ctype = gdb.lookup_type(ctype)
            ptr = ptr.cast(ctype.array(length - 1).pointer()).dereference()
            self.mat = np.array([ptr[i] for i in range(length)], dtype=dtype)
        else:
            # read as uint16_t and then reinterpret the bytes as float16
            u16 = gdb.lookup_type('uint16_t')
            ptr = ptr.cast(u16.array(length - 1).pointer()).dereference()
            self.mat = np.array([ptr[i] for i in range(length)], dtype=np.uint16)
            self.mat = self.mat.view(np.float16)

        # numpy will do the heavy lifting of strided access
        self.view = np.lib.stride_tricks.as_strided(self.mat, shape=shape, strides=steps)

    def __iter__(self):
        return iter({'data': stri(self.view)}.items())


class MatPrinter:
    """Print a cv::Mat"""

    def __init__(self, mat):
        self.mat = mat

    def views(self):
        m = self.mat

        flags = Flags(int(m['flags']))
        size = Size(m['size']['p'])
        data = Mat(m, size, flags)

        for x in [flags, size, data]:
            for k, v in x:
                yield 'view_' + k, v

    def real(self):
        m = self.mat

        for field in m.type.fields():
            k = field.name
            v = m[k]
            yield k, v

        # TODO: add an enum in interface.h with all cv::Mat element types and use that instead
        # yield 'test', gdb.parse_and_eval(f'(cv::MatTypes)0')

    def children(self):  # TODO: hide real members under new child somehow
        yield from self.views()
        yield from self.real()


def get_type(val):
    # Get the type.
    vtype = val.type

    # If it points to a reference, get the reference.
    if vtype.code == gdb.TYPE_CODE_REF:
        vtype = vtype.target()

    # Get the unqualified type, stripped of typedefs.
    vtype = vtype.unqualified().strip_typedefs()

    # Get the type name.
    typename = vtype.tag

    return typename


def mat_printer(val):
    typename = get_type(val)

    if typename is None:
        return None

    if str(typename) == 'cv::Mat':
        return MatPrinter(val)


gdb.pretty_printers.append(mat_printer)
