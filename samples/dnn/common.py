from __future__ import print_function
import hashlib
import time
import os
import sys
import tarfile
import requests
if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

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

class HashMismatchException(Exception):
        def __init__(self, expected, actual):
            Exception.__init__(self)
            self.expected = expected
            self.actual = actual
        def __str__(self):
            return 'Hash mismatch: expected {} vs actual of {}'.format(self.expected, self.actual)

def checkHashsum(expected_sha, filepath, silent=True):
    print('  expected SHA1: {}'.format(expected_sha))
    sha = hashlib.sha1()
    if os.path.exists(filepath):
        print('  there is already a file with the same name')
        with open(filepath, 'rb') as f:
            while True:
                buf = f.read(10*1024*1024)
                if not buf:
                    break
                sha.update(buf)
    actual_sha = sha.hexdigest()
    print('  actual SHA1:{}'.format(actual_sha))
    hashes_matched = expected_sha == actual_sha
    if not hashes_matched and not silent:
        raise HashMismatchException(expected_sha, actual_sha)
    return hashes_matched

def isArchive(filepath):
    return tarfile.is_tarfile(filepath)

class Model:
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.filenames = kwargs.pop('filenames')
        self.loader = kwargs.pop('loader', None)
        self.save_dir = kwargs.pop('save_dir')
        self.shas = kwargs.pop('shas', None)
    

    def __str__(self):
        return 'Model <{}>'.format(self.name)
        
    def get(self):
        print("  Working on " + self.name)
        for filename, sha in zip(self.filenames, self.shas):
            print("  Getting file " + filename)
            filepath = os.path.join(self.save_dir, filename)
            if checkHashsum(sha, filepath):
                print('  hash match - file already exists, skipping')
                continue
            else:
                print('  hash didn\'t match, loading file')

            if not os.path.exists(self.save_dir):
                print('  creating directory: ' + self.save_dir)
                os.makedirs(self.save_dir, exist_ok=True)
            

            print('  hash check failed - loading')
            assert self.loader
            try:
                self.loader.load(filename, self.save_dir)
                print(' done')
                print(' file {}'.format(filename))
                checkHashsum(sha, filepath, silent=False)
            except Exception as e:
                print("  There was some problem with loading file {} for {}".format(filename, self.name))
                print("  Exception: {}".format(e))
                continue

        print("  Finished " + self.name)

class Loader(object):
    MB = 1024*1024
    BUFSIZE = 10*MB
    def __init__(self, download_name, download_sha, archive_member = None):
        self.download_name = download_name
        self.download_sha = download_sha
        self.archive_member = archive_member

    def load(self, requested_file, save_dir):
        filepath = os.path.join(save_dir, self.download_name)
        print("  Preparing to download file " + self.download_name)
        if checkHashsum(self.download_sha, filepath):
            print('  hash match - file already exists, no need to download')
        else:
            filesize = self.download(filepath)
            print('  Downloaded {} with size {} Mb'.format(self.download_name, filesize/self.MB))
            checkHashsum(self.download_sha, filepath, silent=False)
        if self.download_name == requested_file:
            return
        else:
            if isArchive(filepath):
                self.extract(requested_file, filepath, save_dir)
            else:
                raise Exception("Downloaded file has different name")
    
    def download(self, filepath):
        print("Warning: download is not implemented, this is a base class")
        return 0
    
    def extract(self, requested_file, archive_path, save_dir):
        filepath = os.path.join(save_dir, requested_file)
        try:
            with tarfile.open(archive_path) as f:
                if self.archive_member is None:
                    pathDict = dict((os.path.split(elem)[1], os.path.split(elem)[0]) for elem in f.getnames())
                    self.archive_member = pathDict[requested_file]
                assert self.archive_member in f.getnames()
                self.save(filepath, f.extractfile(self.archive_member))
        except Exception as e:
            print('  catch {}'.format(e))
    
    def save(self, filepath, r):
        with open(filepath, 'wb') as f:
            print('  progress ', end='')
            sys.stdout.flush()
            while True:
                buf = r.read(self.BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                print('>', end='')
                sys.stdout.flush()

class URLLoader(Loader):
    def __init__(self, download_name, download_sha, url, archive_member = None):
        super().__init__(download_name, download_sha, archive_member)
        self.download_name = download_name
        self.download_sha = download_sha
        self.url = url

    def download(self, filepath):
        r = urlopen(self.url, timeout=60)
        self.printRequest(r)
        self.save(filepath, r)
        return os.path.getsize(filepath)
    
    def printRequest(self, r):
        def getMB(r):
            d = dict(r.info())
            for c in ['content-length', 'Content-Length']:
                if c in d:
                    return int(d[c]) / self.MB
            return '<unknown>'
        print('  {} {} [{} Mb]'.format(r.getcode(), r.msg, getMB(r)))

class GDriveLoader(Loader):
    BUFSIZE = 1024 * 1024
    PROGRESS_SIZE = 10 * 1024 * 1024
    def __init__(self, download_name, download_sha, gid, archive_member = None):
        super().__init__(download_name, download_sha, archive_member)
        self.download_name = download_name
        self.download_sha = download_sha
        self.gid = gid

    def download(self, filepath):
        session = requests.Session()  # re-use cookies

        URL = "https://docs.google.com/uc?export=download"
        response = session.get(URL, params = { 'id' : self.gid }, stream = True)

        def get_confirm_token(response):  # in case of large files
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None
        token = get_confirm_token(response)

        if token:
            params = { 'id' : self.gid, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        sz = 0
        progress_sz = self.PROGRESS_SIZE
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(self.BUFSIZE):
                if not chunk:
                    continue  # keep-alive

                f.write(chunk)
                sz += len(chunk)
                if sz >= progress_sz:
                    progress_sz += self.PROGRESS_SIZE
                    print('>', end='')
                    sys.stdout.flush()
        print('')
        return sz
