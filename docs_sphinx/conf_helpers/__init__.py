# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

"""Doc-build engine for the OpenCV Sphinx wrapper.

conf.py stays a thin Sphinx-settings file; everything heavy lives here:

* ``state``       — shared config, paths, tag maps, bib/citation numbering,
                    redirect map, anchor indexes, constants
* ``xml_render``  — Doxygen XML -> Markdown primitives (incl. enum synopsis)
* ``stubs``       — API-reference stub writers (groups / classes)
* ``translate``   — Doxygen-flavored .markdown -> MyST (the source-read engine)
* ``patches``     — Sphinx C++ domain / breathe warning patches
* ``postprocess`` — build-finished hook that inlines collaboration-diagram SVGs
* ``build``       — import-time orchestration that populates the shared indexes
"""
