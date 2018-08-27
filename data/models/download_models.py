#!/usr/bin/env python

from __future__ import print_function
import hashlib
import tarfile
import sys
import os
import time
import shutil
import xml.etree.ElementTree as ET
if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

class BufferedReader(object):
    tick = 0
    BUFSIZE = 1 * 1024 * 1024

    def print_progress(self, msg, timeout = 0):
        if time.time() - self.tick > timeout:
            print(msg, end='')
            sys.stdout.flush()
            self.tick = time.time()

    def read(self, in_stream, processing, tag=''):
        self.print_progress(tag)
        while True:
            buf = in_stream.read(self.BUFSIZE)
            if not buf:
                break
            processing(buf)
            self.print_progress('>', 5)
        print(' done')

class HashMismatchException(Exception):
    def __init__(self, expected, actual):
        Exception.__init__(self)
        self.expected = expected[0:7]
        self.actual = actual[0:7]
    def __str__(self):
        return 'Hash mismatch: {} vs {}'.format(self.expected, self.actual)

class HashedItem(object):
    def __init__(self, name, hash):
        self.name = name
        self.hash = hash

    def process(self, body):
        print('FILE {}'.format(self.name))
        try:
            self.verify()
        except Exception as ex:
            print('  {}'.format(ex))
            try:
                body(self.name)
                self.verify()
            except Exception as ex:
                print('  {}'.format(ex))
                print('  FAILURE')
                return False
        return True

    def verify(self):
        sha = hashlib.sha1()
        with open(self.name, 'rb') as file_stream:
            BufferedReader().read(file_stream, sha.update, '  CHECK')
        if self.hash != sha.hexdigest():
            raise HashMismatchException(self.hash, sha.hexdigest())


class MetalinkParser(object):
    NS = {'ml': 'urn:ietf:params:xml:ns:metalink'}

    def __init__(self, fname):
        self.fname = fname

    def create_item(self, elem):
        return HashedItem(elem.attrib['name'], elem.find('ml:hash', self.NS).text)

    def parse(self):
        items = []
        unpack = []
        for file_elem in ET.parse(self.fname).getroot().findall('ml:file', self.NS):
            item = self.create_item(file_elem)
            item.url = file_elem.find('ml:url', self.NS).text
            items.append(item)
            unpack_elem = file_elem.find('ml:unpack', self.NS)
            if unpack_elem is not None and unpack_elem.get('type') == 'tar':
                for unpack_file_elem in unpack_elem.findall('ml:file', self.NS):
                    unpack_item = self.create_item(unpack_file_elem)
                    unpack_item.archive = item.name
                    unpack_item.archive_location = unpack_file_elem.find('ml:location', self.NS).text
                    unpack.append(unpack_item)
        return items, unpack


class Downloader(object):

    @staticmethod
    def getMB(r):
        try:
            sz = r.info()['content-length']
            return '{} Mb'.format(int(sz) / float(1024 * 1024))
        except Exception as e:
            return '<unknown>'

    def download(self, items):
        status = True
        for item in items:
            def body(x):
                print('  {}'.format(item.url))
                with open(x, 'wb') as file_stream:
                    r = urlopen(item.url)
                    BufferedReader().read(r, file_stream.write, '  DL [{}]'.format(Downloader.getMB(r)))
            status = item.process(body) and status
        return status

    def unpack(self, items):
        status = True
        for item in items:
            def body(x):
                with tarfile.open(item.archive) as archive_stream:
                    ex = archive_stream.extractfile(item.archive_location)
                    with open(x, 'wb') as output_stream:
                        BufferedReader().read(ex, output_stream.write, '  UNPACK')
            status = item.process(body) and status
        return status


# Try to copy files from other location (recursive)
class Importer(object):
    def __init__(self, dir):
        self.dir = dir

    def copy(self, items):
        for root, _, filenames in os.walk(self.dir):
            for item in items:
                if item.name in filenames:
                    def body(x):
                        print("  COPY {} ({})".format(x, root))
                        shutil.copy(os.path.join(root, x), x)
                    item.process(body)


if __name__ == '__main__':
    # aria2c --auto-file-renaming=false --conditional-get=true --check-certificate=false --allow-overwrite=true models.meta4
    importer = None
    dir = os.getenv('OPENCV_DNN_TEST_DATA_PATH')
    if dir is not None:
        print("=== Import files from {}".format(dir))
        importer = Importer(dir)

    dirs = []
    for root, _, filenames in os.walk('.'):
        if 'link.meta4' in filenames:
            dirs.append(os.path.abspath(root))

    res = 0
    for root in dirs:
        print("=== Processing directory {}...".format(root))
        os.chdir(root)
        DL_ITEMS, UNPACK_ITEMS = MetalinkParser('link.meta4').parse()
        if importer is not None:
            importer.copy(DL_ITEMS + UNPACK_ITEMS)
        if not (Downloader().download(DL_ITEMS) and Downloader().unpack(UNPACK_ITEMS)):
            res = 1
    sys.exit(res)
