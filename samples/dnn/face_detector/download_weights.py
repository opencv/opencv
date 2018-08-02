#!/usr/bin/env python

from __future__ import print_function
import hashlib
import time
import sys
import xml.etree.ElementTree as ET
if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

class HashMismatchException(Exception):
    def __init__(self, expected, actual):
        Exception.__init__(self)
        self.expected = expected
        self.actual = actual
    def __str__(self):
        return 'Hash mismatch: {} vs {}'.format(self.expected, self.actual)

class MetalinkDownloader(object):
    BUFSIZE = 10*1024*1024
    NS = {'ml': 'urn:ietf:params:xml:ns:metalink'}
    tick = 0

    def download(self, metalink_file):
        status = True
        for file_elem in ET.parse(metalink_file).getroot().findall('ml:file', self.NS):
            url = file_elem.find('ml:url', self.NS).text
            fname = file_elem.attrib['name']
            hash_sum = file_elem.find('ml:hash', self.NS).text
            print('*** {}'.format(fname))
            try:
                self.verify(hash_sum, fname)
            except Exception as ex:
                print('  {}'.format(ex))
                try:
                    print('  {}'.format(url))
                    with open(fname, 'wb') as file_stream:
                        self.buffered_read(urlopen(url), file_stream.write)
                    self.verify(hash_sum, fname)
                except Exception as ex:
                    print('  {}'.format(ex))
                    print('  FAILURE')
                    status = False
                    continue
            print('  SUCCESS')
        return status

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
    sys.exit(0 if MetalinkDownloader().download('weights.meta4') else 1)
