'''
Helper module to download extra data from Internet
'''
from __future__ import print_function
import os
import sys
import yaml
import argparse
import tarfile
import platform
import tempfile
import hashlib
import requests
import shutil
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen
import xml.etree.ElementTree as ET

__all__ = ["downloadFile"]

class HashMismatchException(Exception):
    def __init__(self, expected, actual):
        Exception.__init__(self)
        self.expected = expected
        self.actual = actual
    def __str__(self):
        return 'Hash mismatch: expected {} vs actual of {}'.format(self.expected, self.actual)

def getHashsumFromFile(filepath):
    sha = hashlib.sha1()
    if os.path.exists(filepath):
        print('  there is already a file with the same name')
        with open(filepath, 'rb') as f:
            while True:
                buf = f.read(10*1024*1024)
                if not buf:
                    break
                sha.update(buf)
    hashsum = sha.hexdigest()
    return hashsum

def checkHashsum(expected_sha, filepath, silent=True):
    print('  expected SHA1: {}'.format(expected_sha))
    actual_sha = getHashsumFromFile(filepath)
    print('  actual SHA1:{}'.format(actual_sha))
    hashes_matched = expected_sha == actual_sha
    if not hashes_matched and not silent:
        raise HashMismatchException(expected_sha, actual_sha)
    return hashes_matched

def isArchive(filepath):
    return tarfile.is_tarfile(filepath)

class DownloadInstance:
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.filename = kwargs.pop('filename')
        self.loader = kwargs.pop('loader', None)
        self.save_dir = kwargs.pop('save_dir')
        self.sha = kwargs.pop('sha', None)

    def __str__(self):
        return 'DownloadInstance <{}>'.format(self.name)

    def get(self):
        print("  Working on " + self.name)
        print("  Getting file " + self.filename)
        if self.sha is None:
            print('  No expected hashsum provided, loading file')
        else:
            filepath = os.path.join(self.save_dir, self.sha, self.filename)
            if checkHashsum(self.sha, filepath):
                print('  hash match - file already exists, skipping')
                return filepath
            else:
                print('  hash didn\'t match, loading file')

        if not os.path.exists(self.save_dir):
            print('  creating directory: ' + self.save_dir)
            os.makedirs(self.save_dir)


        print('  hash check failed - loading')
        assert self.loader
        try:
            self.loader.load(self.filename, self.sha, self.save_dir)
            print(' done')
            print(' file {}'.format(self.filename))
            if self.sha is None:
                download_path = os.path.join(self.save_dir, self.filename)
                self.sha = getHashsumFromFile(download_path)
                new_dir = os.path.join(self.save_dir, self.sha)

                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                filepath = os.path.join(new_dir, self.filename)
                if not (os.path.exists(filepath)):
                    shutil.move(download_path, new_dir)
                print('  No expected hashsum provided, actual SHA is {}'.format(self.sha))
            else:
                checkHashsum(self.sha, filepath, silent=False)
        except Exception as e:
            print("  There was some problem with loading file {} for {}".format(self.filename, self.name))
            print("  Exception: {}".format(e))
            return

        print("  Finished " + self.name)
        return filepath

class Loader(object):
    MB = 1024*1024
    BUFSIZE = 10*MB
    def __init__(self, download_name, download_sha, archive_member = None):
        self.download_name = download_name
        self.download_sha = download_sha
        self.archive_member = archive_member

    def load(self, requested_file, sha, save_dir):
        if self.download_sha is None:
            download_dir = save_dir
        else:
            # create a new folder in save_dir to avoid possible name conflicts
            download_dir = os.path.join(save_dir, self.download_sha)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        download_path = os.path.join(download_dir, self.download_name)
        print("  Preparing to download file " + self.download_name)
        if checkHashsum(self.download_sha, download_path):
            print('  hash match - file already exists, no need to download')
        else:
            filesize = self.download(download_path)
            print('  Downloaded {} with size {} Mb'.format(self.download_name, filesize/self.MB))
            if self.download_sha is not None:
                checkHashsum(self.download_sha, download_path, silent=False)
        if self.download_name == requested_file:
            return
        else:
            if isArchive(download_path):
                if sha is not None:
                    extract_dir = os.path.join(save_dir, sha)
                else:
                    extract_dir = save_dir
                if not os.path.exists(extract_dir):
                    os.makedirs(extract_dir)
                self.extract(requested_file, download_path, extract_dir)
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
            print('  progress ', end="")
            sys.stdout.flush()
            while True:
                buf = r.read(self.BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                print('>', end="")
                sys.stdout.flush()

class URLLoader(Loader):
    def __init__(self, download_name, download_sha, url, archive_member = None):
        super(URLLoader, self).__init__(download_name, download_sha, archive_member)
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
        super(GDriveLoader, self).__init__(download_name, download_sha, archive_member)
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

def produceDownloadInstance(instance_name, filename, sha, url, save_dir, download_name=None, download_sha=None, archive_member=None):
    spec_param = url
    loader = URLLoader
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
            loader = GDriveLoader
            spec_param = token
        else:
            print("Warning: possibly wrong Google Drive link")
    return DownloadInstance(
        name=instance_name,
        filename=filename,
        sha=sha,
        save_dir=save_dir,
        loader=loader(download_name, download_sha, spec_param, archive_member)
    )

def getSaveDir():
    env_path = os.environ.get("OPENCV_DOWNLOAD_DATA_PATH", None)
    if env_path:
        save_dir = env_path
    else:
        # TODO reuse binding function cv2.utils.fs.getCacheDirectory when issue #19011 is fixed
        if platform.system() == "Darwin":
            #On Apple devices
            temp_env = os.environ.get("TMPDIR", None)
            if temp_env is None or not os.path.isdir(temp_env):
                temp_dir = Path("/tmp")
                print("Using world accessible cache directory. This may be not secure: ", temp_dir)
            else:
                temp_dir = temp_env
        elif platform.system() == "Windows":
            temp_dir = tempfile.gettempdir()
        else:
            xdg_cache_env = os.environ.get("XDG_CACHE_HOME", None)
            if (xdg_cache_env and xdg_cache_env[0] and os.path.isdir(xdg_cache_env)):
                temp_dir = xdg_cache_env
            else:
                home_env = os.environ.get("HOME", None)
                if (home_env and home_env[0] and os.path.isdir(home_env)):
                    home_path = os.path.join(home_env, ".cache/")
                    if os.path.isdir(home_path):
                        temp_dir = home_path
                else:
                    temp_dir = tempfile.gettempdir()
                    print("Using world accessible cache directory. This may be not secure: ", temp_dir)

        save_dir = os.path.join(temp_dir, "downloads")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def downloadFile(url, sha=None, save_dir=None, filename=None):
    if save_dir is None:
        save_dir = getSaveDir()
    if filename is None:
        filename = "download_" + datetime.now().__str__()
    name = filename
    return produceDownloadInstance(name, filename, sha, url, save_dir).get()

def parseMetalinkFile(metalink_filepath, save_dir):
    NS = {'ml': 'urn:ietf:params:xml:ns:metalink'}
    models = []
    for file_elem in ET.parse(metalink_filepath).getroot().findall('ml:file', NS):
        url = file_elem.find('ml:url', NS).text
        fname = file_elem.attrib['name']
        name = file_elem.find('ml:identity', NS).text
        hash_sum = file_elem.find('ml:hash', NS).text
        models.append(produceDownloadInstance(name, fname, hash_sum, url, save_dir))
    return models

def parseYAMLFile(yaml_filepath, save_dir):
    models = []
    with open(yaml_filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        for name, params in data_loaded.items():
            load_info = params.get("load_info", None)
            if load_info:
                fname = os.path.basename(params.get("model"))
                hash_sum = load_info.get("sha1")
                url = load_info.get("url")
                download_sha = load_info.get("download_sha")
                download_name = load_info.get("download_name")
                archive_member = load_info.get("member")
                models.append(produceDownloadInstance(name, fname, hash_sum, url, save_dir,
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
