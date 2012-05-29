import os, sys, glob, re

sys.path.append("../modules/python/src2/")
sys.path.append("../modules/java/")

import hdr_parser as hp
import rst_parser as rp

rp.show_warnings = False
rp.show_errors = False

allmodules = rp.allmodules
DOCUMENTED_MARKER = "verified"

ERROR_001_NOTACLASS        = 1
ERROR_002_NOTASTRUCT       = 2
ERROR_003_INCORRECTBASE    = 3
ERROR_004_MISSEDNAMESPACE  = 4
ERROR_005_MISSINGPYFUNC    = 5
ERROR_006_INVALIDPYOLDDOC  = 6
ERROR_007_INVALIDPYDOC     = 7
ERROR_008_CFUNCISNOTGLOBAL = 8
ERROR_009_OVERLOADNOTFOUND = 9
ERROR_010_UNKNOWNCLASS     = 10
ERROR_011_UNKNOWNFUNC      = 11

do_python_crosscheck = True
errors_disabled = [ERROR_004_MISSEDNAMESPACE]

doc_signatures_whitelist = [
# templates
"Matx", "Vec", "SparseMat_", "Scalar_", "Mat_", "Ptr", "Size_", "Point_", "Rect_", "Point3_",
"DataType", "detail::RotationWarperBase", "flann::Index_", "CalonderDescriptorExtractor",
"gpu::DevMem2D_", "gpu::PtrStep_", "gpu::PtrElemStep_",
# black boxes
"CvArr", "CvFileStorage",
# other
"InputArray", "OutputArray",
]

defines = ["cvGraphEdgeIdx", "cvFree", "CV_Assert", "cvSqrt", "cvGetGraphVtx", "cvGraphVtxIdx",
"cvCaptureFromFile", "cvCaptureFromCAM", "cvCalcBackProjectPatch", "cvCalcBackProject",
"cvGetHistValue_1D", "cvGetHistValue_2D", "cvGetHistValue_3D", "cvGetHistValue_nD",
"cvQueryHistValue_1D", "cvQueryHistValue_2D", "cvQueryHistValue_3D", "cvQueryHistValue_nD",
# not a real function but behaves as function
"Mat::size",
# ugly "virtual" functions from ml module
"CvStatModel::train", "CvStatModel::predict",
# TODO:
"cvExtractSURF"
]

synonims = {
    "StarDetector" : ["StarFeatureDetector"],
    "MSER" : ["MserFeatureDetector"],
    "GFTTDetector" : ["GoodFeaturesToTrackDetector"],
    "cvCaptureFromFile" : ["cvCreateFileCapture"],
    "cvCaptureFromCAM" : ["cvCreateCameraCapture"],
    "cvCalcArrBackProjectPatch" : ["cvCalcBackProjectPatch"],
    "cvCalcArrBackProject" : ["cvCalcBackProject"],
    "InputArray" : ["_InputArray"],
    "OutputArray" : ["_OutputArray"],
}

if do_python_crosscheck:
    try:
        import cv2
    except ImportError:
        print "Could not load cv2"
        do_python_crosscheck = False

def get_cv2_object(name):
    if name.startswith("cv2."):
        name = name[4:]
    if name.startswith("cv."):
        name = name[3:]
    if name == "Algorithm":
        return cv2.Algorithm__create("Feature2D.ORB"), name
    elif name == "FeatureDetector":
        return cv2.FeatureDetector_create("ORB"), name
    elif name == "DescriptorExtractor":
        return cv2.DescriptorExtractor_create("ORB"), name
    elif name == "BackgroundSubtractor":
        return cv2.BackgroundSubtractorMOG(), name
    elif name == "StatModel":
        return cv2.KNearest(), name
    else:
        return getattr(cv2, name)(), name

def compareSignatures(f, s):
    # function names
    if f[0] != s[0]:
        return False, "name mismatch"
    # return type
    stype = (s[1] or "void")
    ftype = f[1]
    stype = re.sub(r"\b(cv|std)::", "", stype)
    if ftype:
        ftype = re.sub(r"\b(cv|std)::", "", ftype)
    if ftype and ftype != stype:
        return False, "return type mismatch"
    if ("\C" in f[2]) ^ ("\C" in s[2]):
        return False, "const qualifier mismatch"
    if ("\S" in f[2]) ^ ("\S" in s[2]):
        return False, "static qualifier mismatch"
    if ("\V" in f[2]) ^ ("\V" in s[2]):
        return False, "virtual qualifier mismatch"
    if ("\A" in f[2]) ^ ("\A" in s[2]):
        return False, "abstract qualifier mismatch"
    if len(f[3]) != len(s[3]):
        return False, "different number of arguments"
    for idx, arg in enumerate(zip(f[3], s[3])):
        farg = arg[0]
        sarg = arg[1]
        ftype = re.sub(r"\b(cv|std)::", "", (farg[0] or ""))
        stype = re.sub(r"\b(cv|std)::", "", (sarg[0] or ""))
        if ftype != stype:
            return False, "type of argument #" + str(idx+1) + " mismatch"
        fname = farg[1] or "arg" + str(idx)
        sname = sarg[1] or "arg" + str(idx)
        if fname != sname:
            return False, "name of argument #" + str(idx+1) + " mismatch"
        fdef = re.sub(r"\b(cv|std)::", "", (farg[2] or ""))
        sdef = re.sub(r"\b(cv|std)::", "", (sarg[2] or ""))
        if fdef != sdef:
            return False, "default value of argument #" + str(idx+1) + " mismatch"
    return True, "match"

def formatSignature(s):
    _str = ""
    if "/V" in s[2]:
        _str += "virtual "
    if "/S" in s[2]:
        _str += "static "
    if s[1]:
        _str += s[1] + " "
    else:
        if not bool(re.match(r"(\w+\.)*(?P<cls>\w+)\.(?P=cls)", s[0])):
            _str += "void "
    if s[0].startswith("cv."):
        _str += s[0][3:].replace(".", "::")
    else:
        _str += s[0].replace(".", "::")
    if len(s[3]) == 0:
        _str += "()"
    else:
        _str += "( "
        for idx, arg in enumerate(s[3]):
            if idx > 0:
                _str += ", "
            argtype = re.sub(r"\bcv::", "", arg[0])
            bidx = argtype.find('[')
            if bidx < 0:
                _str += argtype + " "
            else:
                _srt += argtype[:bidx]
            if arg[1]:
                _str += arg[1]
            else:
                _str += "arg" + str(idx)
            if bidx >= 0:
                _str += argtype[bidx:]
            if arg[2]:
                _str += "=" + re.sub(r"\bcv::", "", arg[2])
        _str += " )"
    if "/C" in s[2]:
        _str += " const"
    if "/A" in s[2]:
        _str += " = 0"
    return _str


def logerror(code, message, doc = None):
    if code in errors_disabled:
        return
    if doc:
        print doc["file"] + ":" + str(doc["line"]),
    print "error %03d: %s" % (code, message)
    #print

def process_module(module, path):
    hppparser = hp.CppHeaderParser()
    rstparser = rp.RstParser(hppparser)

    rstparser.parse(module, path)
    rst = rstparser.definitions

    hdrlist = glob.glob(os.path.join(path, "include", "opencv2", module, "*.h*"))
    hdrlist.extend(glob.glob(os.path.join(path, "include", "opencv2", module, "detail", "*.h*")))

    if module == "gpu":
        hdrlist.append(os.path.join(path, "..", "core", "include", "opencv2", "core", "devmem2d.hpp"))
        hdrlist.append(os.path.join(path, "..", "core", "include", "opencv2", "core", "gpumat.hpp"))

    decls = []
    for hname in hdrlist:
        if not "ts_gtest.h" in hname:
            decls += hppparser.parse(hname, wmode=False)

    funcs = []
    # not really needed to hardcode all the namespaces. Normally all they are collected automatically
    namespaces = ['cv', 'cv.gpu', 'cvflann', 'cvflann.anyimpl', 'cvflann.lsh', 'cv.flann', 'cv.linemod', 'cv.detail', 'cvtest', 'perf', 'cv.videostab']
    classes = []
    structs = []

    # collect namespaces and classes/structs
    for decl in decls:
        if decl[0].startswith("const"):
            pass
        elif decl[0].startswith("class") or decl[0].startswith("struct"):
            if decl[0][0] == 'c':
                classes.append(decl)
            else:
                structs.append(decl)
            dotIdx = decl[0].rfind('.')
            if dotIdx > 0:
                namespace = decl[0][decl[0].find(' ')+1:dotIdx]
                if not [c for c in classes if c[0].endswith(namespace)] and not [s for s in structs if s[0].endswith(namespace)]:
                    if namespace not in namespaces:
                        namespaces.append(namespace)
        else:
            funcs.append(decl)

    clsnamespaces = []
    # process classes
    for cl in classes:
        name = cl[0][cl[0].find(' ')+1:]
        if name.find('.') < 0 and not name.startswith("Cv"):
            logerror(ERROR_004_MISSEDNAMESPACE, "class " + name + " from opencv_" + module + " is placed in global namespace but violates C-style naming convention")
        clsnamespaces.append(name)
        if do_python_crosscheck and not name.startswith("cv.") and name.startswith("Cv"):
            clsnamespaces.append("cv." + name[2:])
        if name.startswith("cv."):
            name = name[3:]
        name = name.replace(".", "::")
        sns = synonims.get(name, [])
        sns.append(name)
        for name in sns:
            doc = rst.get(name)
            if not doc:
                #TODO: class is not documented
                continue
            doc[DOCUMENTED_MARKER] = True
            # verify class marker
            if not doc.get("isclass"):
                logerror(ERROR_001_NOTACLASS, "class " + name + " is not marked as \"class\" in documentation", doc)
            else:
                # verify base
                signature = doc.get("class", "")
                signature = signature.replace(", public ", " ").replace(" public ", " ")
                signature = signature.replace(", protected ", " ").replace(" protected ", " ")
                signature = signature.replace(", private ", " ").replace(" private ", " ")
                signature = ("class " + signature).strip()
                #hdrsignature = (cl[0] + " " +  cl[1]).replace("class cv.", "class ").replace(".", "::").strip()
                hdrsignature = ("class " + name + " " +  cl[1]).replace(".", "::").strip()
                if signature != hdrsignature:
                    logerror(ERROR_003_INCORRECTBASE, "invalid base class documentation\ndocumented: " + signature + "\nactual:     " + hdrsignature, doc)

    # process structs
    for st in structs:
        name = st[0][st[0].find(' ')+1:]
        if name.find('.') < 0 and not name.startswith("Cv"):
            logerror(ERROR_004_MISSEDNAMESPACE, "struct " + name + " from opencv_" + module + " is placed in global namespace but violates C-style naming convention")
        clsnamespaces.append(name)
        if name.startswith("cv."):
            name = name[3:]
        name = name.replace(".", "::")
        doc = rst.get(name)
        if not doc:
            #TODO: struct is not documented
            continue
        doc[DOCUMENTED_MARKER] = True
        # verify struct marker
        if not doc.get("isstruct"):
            logerror(ERROR_002_NOTASTRUCT, "struct " + name + " is not marked as \"struct\" in documentation", doc)
        else:
            # verify base
            signature = doc.get("class", "")
            signature = signature.replace(", public ", " ").replace(" public ", " ")
            signature = signature.replace(", protected ", " ").replace(" protected ", " ")
            signature = signature.replace(", private ", " ").replace(" private ", " ")
            signature = ("struct " + signature).strip()
            hdrsignature = (st[0] + " " +  st[1]).replace("struct cv.", "struct ").replace(".", "::").strip()
            if signature != hdrsignature:
                logerror(ERROR_003_INCORRECTBASE, "invalid base struct documentation\ndocumented: " + signature + "\nactual:     " + hdrsignature, doc)
                print st, doc

    # process functions and methods
    flookup = {}
    for fn in funcs:
        name = fn[0]
        parent = None
        namespace = None
        for cl in clsnamespaces:
            if name.startswith(cl + "."):
                if cl.startswith(parent or ""):
                    parent = cl
        if parent:
            name = name[len(parent) + 1:]
            for nm in namespaces:
                if parent.startswith(nm + "."):
                    if nm.startswith(namespace or ""):
                        namespace = nm
            if namespace:
                parent = parent[len(namespace) + 1:]
        else:
            for nm in namespaces:
                if name.startswith(nm + "."):
                    if nm.startswith(namespace or ""):
                        namespace = nm
            if namespace:
                name = name[len(namespace) + 1:]
        #print namespace, parent, name, fn[0]
        if not namespace and not parent and not name.startswith("cv") and not name.startswith("CV_"):
            logerror(ERROR_004_MISSEDNAMESPACE, "function " + name + " from opencv_" + module + " is placed in global namespace but violates C-style naming convention")
        else:
            fdescr = (namespace, parent, name, fn)
            flookup_entry = flookup.get(fn[0], [])
            flookup_entry.append(fdescr)
            flookup[fn[0]] = flookup_entry

    if do_python_crosscheck:
        for name, doc in rst.iteritems():
            decls = doc.get("decls")
            if not decls:
                continue
            for signature in decls:
                if signature[0] == "Python1":
                    pname = signature[1][:signature[1].find('(')]
                    try:
                        fn = getattr(cv2.cv, pname[3:])
                        docstr = "cv." + fn.__doc__
                    except AttributeError:
                        logerror(ERROR_005_MISSINGPYFUNC, "could not load documented function: cv2." + pname, doc)
                        continue
                    docstring = docstr
                    sign = signature[1]
                    signature.append(DOCUMENTED_MARKER)
                    # convert old signature to pydoc style
                    if docstring.endswith("*"):
                        docstring = docstring[:-1]
                    s = None
                    while s != sign:
                        s = sign
                        sign = re.sub(r"^(.*\(.*)\(.*?\)(.*\) *->)", "\\1_\\2", sign)
                    s = None
                    while s != sign:
                        s = sign
                        sign = re.sub(r"\s*,\s*([^,]+)\s*=\s*[^,]+\s*(( \[.*\])?)\)", " [, \\1\\2])", sign)
                    sign = re.sub(r"\(\s*([^,]+)\s*=\s*[^,]+\s*(( \[.*\])?)\)", "([\\1\\2])", sign)

                    sign = re.sub(r"\)\s*->\s*", ") -> ", sign)
                    sign = sign.replace("-> convexHull", "-> CvSeq")
                    sign = sign.replace("-> lines", "-> CvSeq")
                    sign = sign.replace("-> boundingRects", "-> CvSeq")
                    sign = sign.replace("-> contours", "-> CvSeq")
                    sign = sign.replace("-> retval", "-> int")
                    sign = sign.replace("-> detectedObjects", "-> CvSeqOfCvAvgComp")

                    def retvalRplace(match):
                        m = match.group(1)
                        m = m.replace("CvScalar", "scalar")
                        m = m.replace("CvMemStorage", "memstorage")
                        m = m.replace("ROIplImage", "image")
                        m = m.replace("IplImage", "image")
                        m = m.replace("ROCvMat", "mat")
                        m = m.replace("CvMat", "mat")
                        m = m.replace("double", "float")
                        m = m.replace("CvSubdiv2DPoint", "point")
                        m = m.replace("CvBox2D", "Box2D")
                        m = m.replace("IplConvKernel", "kernel")
                        m = m.replace("CvHistogram", "hist")
                        m = m.replace("CvSize", "width,height")
                        m = m.replace("cvmatnd", "matND")
                        m = m.replace("CvSeqOfCvConvexityDefect", "convexityDefects")
                        mm = m.split(',')
                        if len(mm) > 1:
                            return "(" + ", ".join(mm) + ")"
                        else:
                            return m

                    docstring = re.sub(r"(?<=-> )(.*)$", retvalRplace, docstring)
                    docstring = docstring.replace("( [, ", "([")

                    if sign != docstring:
                        logerror(ERROR_006_INVALIDPYOLDDOC, "old-style documentation differs from pydoc\npydoc: " + docstring + "\nfixup: " + sign + "\ncvdoc: " + signature[1], doc)
                elif signature[0] == "Python2":
                    pname = signature[1][4:signature[1].find('(')]
                    cvname = "cv." + pname
                    parent = None
                    for cl in clsnamespaces:
                        if cvname.startswith(cl + "."):
                            if cl.startswith(parent or ""):
                                parent = cl
                    try:
                        if parent:
                            instance, clsname = get_cv2_object(parent)
                            fn = getattr(instance, cvname[len(parent)+1:])
                            docstr = fn.__doc__
                            docprefix = "cv2." + clsname + "."
                        else:
                            fn = getattr(cv2, pname)
                            docstr = fn.__doc__
                            docprefix = "cv2."
                    except AttributeError:
                        if parent:
                            logerror(ERROR_005_MISSINGPYFUNC, "could not load documented member of " + parent + " class: cv2." + pname, doc)
                        else:
                            logerror(ERROR_005_MISSINGPYFUNC, "could not load documented function cv2." + pname, doc)
                        signature.append(DOCUMENTED_MARKER) # stop subsequent errors
                        continue
                    docstrings = [docprefix + s.replace("([, ", "([") for s in docstr.split("  or  ")]
                    if not signature[1] in docstrings:
                        pydocs = "\npydoc: ".join(docstrings)
                        logerror(ERROR_007_INVALIDPYDOC, "documentation differs from pydoc\npydoc: " + pydocs + "\ncvdoc: " + signature[1], doc)
                    signature.append(DOCUMENTED_MARKER)

    # verify C/C++ signatures
    for name, doc in rst.iteritems():
        decls = doc.get("decls")
        if not decls:
            continue
        for signature in decls:
            if signature[0] == "C" or signature[0] == "C++":
                if "template" in (signature[2][1] or ""):
                    # TODO find a way to validate templates
                    signature.append(DOCUMENTED_MARKER)
                    continue
                fd = flookup.get(signature[2][0])
                if not fd:
                    if signature[2][0].startswith("cv."):
                        fd = flookup.get(signature[2][0][3:])
                    if not fd:
                        continue
                    else:
                        signature[2][0] = signature[2][0][3:]
                if signature[0] == "C":
                    ffd = [f for f in fd if not f[0] and not f[1]] # filter out C++ stuff
                    if not ffd:
                        if fd[0][1]:
                            logerror(ERROR_008_CFUNCISNOTGLOBAL, "function " + fd[0][2] + " is documented as C function but is actually member of " + fd[0][1] + " class", doc)
                        elif fd[0][0]:
                            logerror(ERROR_008_CFUNCISNOTGLOBAL, "function " + fd[0][2] + " is documented as C function but is actually placed in " + fd[0][0] + " namespace", doc)
                    fd = ffd
                error = None
                for f in fd:
                    match, error = compareSignatures(signature[2], f[3])
                    if match:
                        signature.append(DOCUMENTED_MARKER)
                        break
                if signature[-1] != DOCUMENTED_MARKER:
                    candidates = "\n\t".join([formatSignature(f[3]) for f in fd])
                    logerror(ERROR_009_OVERLOADNOTFOUND, signature[0] + " function " + signature[2][0].replace(".","::") + " is documented but misses in headers (" + error + ").\nDocumented as:\n\t" + signature[1] + "\nCandidates are:\n\t" + candidates, doc)
                    signature.append(DOCUMENTED_MARKER) # to stop subsequent error on this function

    # verify that all signatures was found in the library headers
    for name, doc in rst.iteritems():
        # if doc.get(DOCUMENTED_MARKER, False):
        #     continue # this class/struct was found
        if not doc.get(DOCUMENTED_MARKER, False) and (doc.get("isclass", False) or doc.get("isstruct", False)):
            if name in doc_signatures_whitelist:
                continue
            logerror(ERROR_010_UNKNOWNCLASS, "class/struct " + name + " is mentioned in documentation but is not found in OpenCV headers", doc)
        for d in doc.get("decls", []):
            if d[-1] != DOCUMENTED_MARKER:
                if d[0] == "C" or d[0] =="C++" or (do_python_crosscheck and d[0].startswith("Python")):
                    if d[0][0] == 'C':
                        sname = d[2][0][3:].replace(".", "::")
                        if sname in defines:
                            #TODO: need to find a way to verify #define's
                            continue
                    else:
                        sname = d[1][:d[1].find("(")]
                    prefixes = [x for x in doc_signatures_whitelist if sname.startswith(x)]
                    if prefixes:
                        # TODO: member of template class
                        continue
                    logerror(ERROR_011_UNKNOWNFUNC, d[0] + " function " + sname + " is documented but is not found in OpenCV headers. It is documented as:\n\t" + d[1], doc)
    # end of process_module

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage:\n", os.path.basename(sys.argv[0]), " <module path>"
        exit(0)

    modules = sys.argv[1:]
    if modules[0] == "all":
        modules = allmodules

    for module in modules:
        selfpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        module_path = os.path.join(selfpath, "..", "modules", module)

        if not os.path.isdir(module_path):
            print "Module \"" + module + "\" could not be found."
            exit(1)

        process_module(module, module_path)
