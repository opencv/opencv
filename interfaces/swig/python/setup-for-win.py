from distutils.core import setup, Extensionimport os
opencv_pwrap_dir = r'.'
opencv_base_dir = r'../../..'

def patch_for_win32(filename,outfile,patches,extra_defs):
    print 'patching '+filename+'...'
    src = open(filename,'rt')
    dst = open(outfile, 'wt')
    for l in src.xreadlines():
        dl = l
        for (from_str,to_str) in patches:
            dl = dl.replace(from_str,to_str)
        for i in extra_defs:
            if l.find(i[0]) >= 0:
                dst.write(i[1])
                extra_defs.remove(i)
        dst.write(dl)
    src.close()
    dst.close()

def is_older(a,b):
    return os.path.getmtime(a)<os.path.getmtime(b)

if not os.path.exists('_cv_win32.cpp') or is_older('_cv_win32.cpp','_cv.cpp'):
    patch_for_win32('_cv.cpp', '_cv_win32.cpp',
        [('unsigned long long','uint64',),('long long','int64'),
        ("char *doc = (((PyCFunctionObject *)obj) -> m_ml -> ml_doc);",
        "char *doc = (char*)(((PyCFunctionObject *)obj) -> m_ml -> ml_doc);"),
        ("char *c = methods[i].ml_doc;",
        "char *c = (char*)methods[i].ml_doc;")],
        [('PyAPI_FUNC','#undef PyAPI_FUNC\n'), ('cv.h',
"""
#include "cv.h"

const signed char icvDepthToType[]=
{
    -1, -1, CV_8U, CV_8S, CV_16U, CV_16S, -1, -1,
    CV_32F, CV_32S, -1, -1, -1, -1, -1, -1, CV_64F, -1
};

CvModuleInfo* CvModule::first = 0;
CvModuleInfo* CvModule::last = 0;
CvTypeInfo* CvType::first = 0;
CvTypeInfo* CvType::last = 0;

""")])

if not os.path.exists('_highgui_win32.cpp') or is_older('_highgui_win32.cpp','_highgui.cpp'):
    patch_for_win32('_highgui.cpp', '_highgui_win32.cpp',
        [('unsigned long long','uint64',),('long long','int64'),
        ("char *doc = (((PyCFunctionObject *)obj) -> m_ml -> ml_doc);",
        "char *doc = (char*)(((PyCFunctionObject *)obj) -> m_ml -> ml_doc);"),
        ("char *c = methods[i].ml_doc;",
        "char *c = (char*)methods[i].ml_doc;")],
        [('PyAPI_FUNC','#undef PyAPI_FUNC\n')])

if not os.path.exists('_ml_win32.cpp') or is_older('_ml_win32.cpp','_ml.cpp'):
    patch_for_win32('_ml.cpp', '_ml_win32.cpp',
        [('unsigned long long','uint64',),('long long','int64'),
        ("char *doc = (((PyCFunctionObject *)obj) -> m_ml -> ml_doc);",
        "char *doc = (char*)(((PyCFunctionObject *)obj) -> m_ml -> ml_doc);"),
        ("char *c = methods[i].ml_doc;",
        "char *c = (char*)methods[i].ml_doc;")],
        [('PyAPI_FUNC','#undef PyAPI_FUNC\n')])


setup(name='OpenCV Python Wrapper',
      version='0.0',
      packages = ['opencv'],
      package_dir = {'opencv': opencv_pwrap_dir},
      ext_modules=[Extension('opencv._cv',
                             [os.path.join (opencv_pwrap_dir, '_cv_win32.cpp'),
                              os.path.join (opencv_pwrap_dir, 'error.cpp'),
                              os.path.join (opencv_pwrap_dir, 'cvshadow.cpp'),
                              os.path.join (opencv_pwrap_dir, 'pyhelpers.cpp')],
                             include_dirs = [os.path.join (opencv_base_dir,
                                                           'cv', 'include'),
                                             os.path.join (opencv_base_dir,
                                                           'cxcore', 'include'),
                                                           ],
                             library_dirs = [os.path.join (opencv_base_dir,
                                                           'lib')],
                             libraries = ['cv', 'cxcore'],
                             ),

                   Extension('opencv._ml',
                             [os.path.join (opencv_pwrap_dir, '_ml_win32.cpp'),
                              os.path.join (opencv_pwrap_dir, 'error.cpp'),
                              os.path.join (opencv_pwrap_dir, 'cvshadow.cpp'),
                              os.path.join (opencv_pwrap_dir, 'pyhelpers.cpp')],
                             include_dirs = [os.path.join (opencv_base_dir,
                                                           'cv', 'include'),
                                             os.path.join (opencv_base_dir,
                                                           'cxcore', 'include'),
                                             os.path.join (opencv_base_dir,
                                                           'ml', 'include'),
                                             os.path.join (opencv_base_dir,
                                                           'otherlibs', 'highgui'),
                                                           ],
                             library_dirs = [os.path.join (opencv_base_dir,
                                                           'lib')],
                             libraries = ['cv', 'cxcore', 'ml'],
                             ),

                   Extension('opencv._highgui',                             [os.path.join (opencv_pwrap_dir, '_highgui_win32.cpp'),
                              os.path.join (opencv_pwrap_dir, 'error.cpp'),
                              os.path.join (opencv_pwrap_dir, 'cvshadow.cpp'),
                              os.path.join (opencv_pwrap_dir, 'pyhelpers.cpp')],
                             include_dirs = [os.path.join (opencv_base_dir,
                                                           'otherlibs', 'highgui'),
                                             os.path.join (opencv_base_dir,
                                                           'cxcore', 'include'),
                                             os.path.join (opencv_base_dir,
                                                           'cv', 'include')],
                             library_dirs = [os.path.join (opencv_base_dir,
                                                           'lib')],
                             libraries = ['highgui', 'cv', 'cxcore'],
                             )
                   ]
      )
