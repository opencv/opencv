#!/usr/bin/python
# Copyright 2008 Emanuele Rocca <ema@galliera.it>
# Copyright 2008 Marco De Benedetto <debe@galliera.it>
# Copyright (c) 2006-2011 Mathieu Malaterre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A WADO server, compliant with section 3.18 of the DICOM standard.

See http://www.dclunie.com/dicom-status/status.html for more information about
DICOM.

Basic usage example:

requestType=WADO&studyUID=1.3.76.13.10010.0.5.74.6120.1224052386.1578&seriesUID=1.3.76.2.1.1.3.1.3.3115.274695982&objectUID=1.3.76.2.1.1.3.1.3.3115.274695982.26&contentType=image/png
"""

import cherrypy

from flup.server.fcgi import WSGIServer

import sys
import os

CURDIR = os.path.dirname(os.path.abspath(__file__))

import accessdata

try:
    testing = sys.argv[1] == "test"
    BASEURL = "/"
except IndexError:
    BASEURL = "/wado/"
    testing = False

REQUIRED = ( 'requestType', 'studyUID', 'seriesUID', 'objectUID' )

OPTIONAL = ( 'contentType', 'charset', 'anonymize', 'annotation',
             'rows', 'columns', 'region',
             'windowWidth', 'windowCenter', 'frameNumber', 'imageQuality',
             'presentationUID', 'presentationSeriesUID', 'transferSyntax' )

INVALID_DICOM = ( 'annotation', 'rows', 'columns', 'region',
                  'windowWidth', 'windowCenter' )

INVALID_NONDICOM = ( 'anonymize', )

def err(msg):
    """Function to handle errors"""
    raise Exception, msg

def check_params(kwargs):
    """Validate and sanitize requests"""
    # TODO: implement every check
    valid = REQUIRED + OPTIONAL

    curparams = kwargs.keys()

    # WADO is the only requestType currently accepted by the standard
    assert kwargs['requestType'] == "WADO"

    # checking unknown parameters
    for par in curparams:
        if par not in valid:
            err("Unknown parameter: " + par)

    # checking missing parameters
    for par in REQUIRED:
        if par not in curparams:
            err("Missing parameter: " + par)

    # default content type is image/jpeg
    kwargs['contentType'] = kwargs.get('contentType', 'image/jpeg')

    # checking values for contentType = application/dicom
    if kwargs['contentType'] == 'application/dicom':
        for par in INVALID_DICOM:
            if par in curparams:
                err(par + " is not valid if contentType is application/dicom")

        # Validation finished
        return

    # checking values for contentType != application/dicom
    for par in INVALID_NONDICOM:
        if par in curparams:
            err(par + " is valid only if contentType is application/dicom")

    if 'annotation' in curparams:
        assert kwargs['annotation'] in ('patient', 'technique')

    if 'windowWidth' in curparams:
        assert 'windowCenter' in curparams

    if 'windowCenter' in curparams:
        assert 'windowWidth' in curparams

    if 'region' in curparams:
        region = kwargs['region'].split(',')

        assert len(region) == 4

        for val in region:
            assert 0.0 <= float(val) <= 1.0

class Wado:
    """Wado controller"""

    @cherrypy.expose
    def index(self, **kwargs):
        cherrypy.log(str(kwargs))

        check_params(kwargs)

        cherrypy.response.headers['Content-Type'] = kwargs['contentType']
        cherrypy.response.headers['Pragma'] = 'cache'

        if kwargs['contentType'] == "application/dicom":
            format = "dicom"
        else:
            # image/png -> png, image/jpeg -> jpeg
            format = kwargs['contentType'].replace('image/', '')

        # getting DICOM image from accessdata
        image = accessdata.get(studyUID=kwargs['studyUID'],
                               seriesUID=kwargs['seriesUID'],
                               objectUID=kwargs['objectUID'],
                               format=format)

        if kwargs['contentType'] == "application/dicom":
            return image.raw()

        if 'windowWidth' in kwargs:
            image.brightness(kwargs['windowWidth'])

        if 'windowCenter' in kwargs:
            image.contrast(kwargs['windowCenter'])

        if 'region' in kwargs:
            left, upper, right, lower = [
                float(val) for val in kwargs['region'].split(",")
            ]
            # coordinates normalization
            width, height = image.img.size
            # 1 : width = left : x
            image.crop(width * left, height * upper,
                       width * right, height * lower)

        if 'rows' in kwargs or 'columns' in kwargs:
            image.resize(kwargs['rows'], kwargs['columns'])

        return image.dump()


if __name__ == "__main__":
    cherrypy.config.update({
        #'server.socket_port': 7777,
        #'server.thread_pool': 1,
        #'environment': testing and 'testing' or 'production',
        #'log.access_file': 'access.log',
        #'log.error_file': 'errors.log',
        #'tools.sessions.on': True,
        'tools.gzip.on': True,
        'tools.caching.on': True
    })

    conf = {}

    app = cherrypy.tree.mount(Wado(), BASEURL, config=conf)

    if testing:
        cherrypy.quickstart(app)
    else:
        cherrypy.engine.start(blocking=False)
        try:
            WSGIServer(app).run()
        finally:
            cherrypy.engine.stop()
