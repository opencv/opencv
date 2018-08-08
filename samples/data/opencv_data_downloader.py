#!/usr/bin/env python
'''
Helper module to download extra data from Internet
'''
from __future__ import print_function
import hashlib
import os
import sys
import time
import xml.etree.ElementTree as ET
PY3 = sys.version_info[0] == 3
if PY3:
    from urllib.request import urlopen
    long = int
else:
    from urllib2 import urlopen

class HashMismatchException(Exception):
    def __init__(self, expected, actual):
        Exception.__init__(self)
        self.expected = expected
        self.actual = actual
    def __str__(self):
        return 'Hash mismatch: {} vs {}'.format(self.expected, self.actual)

class Downloader(object):
    BUFSIZE = 64*1024
    NS = {'ml': 'urn:ietf:params:xml:ns:metalink'}
    tick = 0

    def download_metalink(self, metalink_file, dst_path=None):
        status = True
        for file_elem in ET.parse(metalink_file).getroot().findall('ml:file', self.NS):
            url = file_elem.find('ml:url', self.NS).text
            fname = file_elem.attrib['name']
            hash_sum = file_elem.find('ml:hash', self.NS).text
            size = long(file_elem.find('ml:size', self.NS).text)
            if not self.download_file(fname, url, hash_sum, size=size, dst_path=dst_path):
                status = False
        return status

    def verify_file(self, fname, hash_sum):
        if not os.path.exists(fname):
            return False
        try:
            print('  Check file contents ...')
            self.verify(hash_sum, fname)
            return True
        except Exception as ex:
            print('  Exception: {}'.format(ex))
            return False

    def download_file(self, fname, url, hash_sum, size=None, dst_path=None):
        try:
            current_dir = os.getcwd()
            if dst_path is not None:
                print('*** Download into directory: {}'.format(dst_path))
                os.chdir(dst_path)
            print('*** {}  size={:.3f} Mb'.format(fname, size / (1024*1024.0)))
            if not self.verify_file(fname, hash_sum):
                try:
                    print('  Downloading: {} ...'.format(url))
                    with open(fname, 'wb') as file_stream:
                        self.buffered_read(urlopen(url), file_stream.write)
                    print('  Verifying ...')
                    self.verify(hash_sum, fname)
                except Exception as ex:
                    print('  Exception: {}'.format(ex))
                    print('  FAILURE')
                    return False
            print('  SUCCESS')
            return True
        finally:
            os.chdir(current_dir)  # restore path

    def print_progress(self, msg, timeout = 0):
        if time.time() - self.tick > timeout:
            print(msg, end='')
            sys.stdout.flush()
            self.tick = time.time()

    def buffered_read(self, in_stream, processing):
        self.print_progress('  >')
        while True:
            buf = in_stream.read(self.BUFSIZE)
            if not buf:
                break
            processing(buf)
            self.print_progress('>', 5)
        print(' done')

    def verify(self, hash_sum, fname):
        sha = hashlib.sha1()
        with open(fname, 'rb') as file_stream:
            self.buffered_read(file_stream, sha.update)
        if hash_sum != sha.hexdigest():
            raise HashMismatchException(hash_sum, sha.hexdigest())

if __name__ == '__main__':
    print("This is utility module. Don't use it as standalone application")
