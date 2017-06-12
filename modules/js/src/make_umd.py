#!/usr/bin/python
import os, sys, re, json, shutil
from subprocess import Popen, PIPE, STDOUT

def make_umd(opencvjs, cvjs):
    src = open(opencvjs, 'r+b')
    dst = open(cvjs, 'w+b')
    content = src.read()
    dst.seek(0)
    # inspired by https://github.com/umdjs/umd/blob/95563fd6b46f06bda0af143ff67292e7f6ede6b7/templates/returnExportsGlobal.js
    dst.write(("""
(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(function () {
      return (root.cv = factory());
    });
  } else if (typeof module === 'object' && module.exports) {
    // Node. Does not work with strict CommonJS, but
    // only CommonJS-like environments that support module.exports,
    // like Node.
    module.exports = factory();
  } else {
    // Browser globals
    root.cv = factory();
  }
}(this, function () {
  %s
  return cv(Module);
}));
    """ % (content)).lstrip())

if __name__ == "__main__":
    if len(sys.argv) > 2:
        opencvjs = sys.argv[1]
        cvjs = sys.argv[2]
        make_umd(opencvjs, cvjs);
