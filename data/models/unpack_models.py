#!/usr/bin/env python

import sys
from download_models import MetalinkParser, Downloader

if __name__ == '__main__':
    _, UNPACK_ITEMS = MetalinkParser('link.meta4').parse()
    sys.exit(0 if Downloader().unpack(UNPACK_ITEMS) else 1)
