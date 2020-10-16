'''
Helper module to download extra data from Internet
'''
import os
import sys
import yaml
import argparse
import tarfile
import hashlib
import requests
if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen
import xml.etree.ElementTree as ET

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

def produceModel(model_name, filename, sha, url, save_dir, download_name=None, download_sha=None, archive_member=None):
    if download_name is None:
        download_name = filename
    if download_sha is None:
        download_sha = sha
    if "drive.google.com" in url:
        token = ""
        token_part = url.rsplit('/', 1)[-1]
        if "&id=" not in token_part:
            token_part = url.rsplit('/', 1)[-2]
        for param in token_part.split("&"):
            if param.startswith("id="):                
                token = param[3:]
        if token:
            loader = GDriveLoader(download_name, download_sha, token, archive_member)
        else:
            print("Warning: possibly wrong Google Drive link")
            loader = URLLoader(download_name, download_sha, url, archive_member)
    else:
        loader = URLLoader(download_name, download_sha, url, archive_member)
    return Model(
        name=model_name,
        filenames=[filename],
        shas=[sha],
        save_dir=save_dir,
        loader=loader
    )

def parseMetalinkFile(metalink_filepath, save_dir):
    NS = {'ml': 'urn:ietf:params:xml:ns:metalink'}
    models = []
    for file_elem in ET.parse(metalink_filepath).getroot().findall('ml:file', NS):
        url = file_elem.find('ml:url', NS).text
        fname = file_elem.attrib['name']
        name = file_elem.find('ml:identity', NS).text
        hash_sum = file_elem.find('ml:hash', NS).text
        models.append(produceModel(name, fname, hash_sum, url, save_dir))
    return models

def parseYAMLFile(yaml_filepath, save_dir):
    models = []
    with open(yaml_filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        for name, params in data_loaded.items():
            load_info = params.get("load_info", None)
            if load_info:
                fname = os.path.basename(params.get("model"))
                hash_sum = load_info.get("sha")
                url = load_info.get("url")
                download_sha = load_info.get("download_sha")
                download_name = load_info.get("download_name")
                archive_member = load_info.get("member")
                models.append(produceModel(name, fname, hash_sum, url, save_dir, 
                    download_name=download_name, download_sha=download_sha, archive_member=archive_member))

    return models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is a utility script for downloading DNN models for samples.')

    parser.add_argument('--save_dir', action="store", default=os.getcwd(),
                        help='Path to the directory to store downloaded files')
    parser.add_argument('model_name', type=str, default="", nargs='?', action="store",
                        help='name of the model to download')
    args = parser.parse_args()
    models = []
    save_dir = args.save_dir
    selected_model_name = args.model_name
    models.extend(parseMetalinkFile('face_detector/weights.meta4', save_dir))
    models.extend(parseYAMLFile('models.yml', save_dir))
    for m in models:
        print(m)
        if selected_model_name and not m.name.startswith(selected_model_name):
            continue
        print('Model: ' + selected_model_name)
        m.get()