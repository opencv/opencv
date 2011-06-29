import sys, glob

sys.path.append("../modules/python/src2/")
import hdr_parser as hp

opencv_hdr_list = [
"../modules/core/include/opencv2/core/core.hpp",
"../modules/ml/include/opencv2/ml/ml.hpp",
"../modules/imgproc/include/opencv2/imgproc/imgproc.hpp",
"../modules/calib3d/include/opencv2/calib3d/calib3d.hpp",
"../modules/features2d/include/opencv2/features2d/features2d.hpp",
"../modules/video/include/opencv2/video/tracking.hpp",
"../modules/video/include/opencv2/video/background_segm.hpp",
"../modules/objdetect/include/opencv2/objdetect/objdetect.hpp",
"../modules/highgui/include/opencv2/highgui/highgui.hpp",
]

opencv_module_list = [
"core",
"imgproc",
"calib3d",
"features2d",
"video",
"objdetect",
"highgui",
"ml"
]

class RSTParser(object):
    
    def __init__(self):
        self.read_whitelist()
    
    # reads the file containing functions and classes that do not need to be documented
    def read_whitelist(self):
        self.whitelist = {}
        try:
            wf = open("check_docs_whitelist.txt", "rt")
        except IOError:
            return
        self.parser = hp.CppHeaderParser()
        
        for l in wf.readlines():
            cpos = l.find("#")
            if cpos >= 0:
                l = l[:cpos]
            l = l.strip()
            if not l:
                continue
            rst_decl = None
            if "(" in l:
                l = l.replace("cv::", "")
                rst_decl = self.parser.parse_func_decl_no_wrap(l)
                fname = rst_decl[0]
            else:
                fname = l.replace("::", ".")
            complist = fname.split(".")
            prefix = ""
            alreadyListed = False
            wl = []
            for c in complist:
                prefix = (prefix + "." + c).lstrip(".")
                wl = self.whitelist.get(prefix, [])
                if wl == "*":
                    break
            if wl == "*":
                continue
            if not rst_decl:
                self.whitelist[fname] = "*"
            else:
                wl.append(rst_decl)
                self.whitelist[fname] = wl
        wf.close()
            
    def process_rst(self, docname):
        df = open(docname, "rt")
        fdecl = ""
        balance = 0
        lineno = 0
        
        for l in df.readlines():
            lineno += 1
            ll = l.strip()
            if balance == 0:
                if not ll.startswith(".. c:function::") and \
                   not ll.startswith(".. cpp:function::") and \
                   not ll.startswith(".. ocv:function::") and \
                   not ll.startswith(".. ocv:cfunction::"):
                    continue
                fdecl = ll[ll.find("::") + 3:]
            elif balance > 0:
                fdecl += ll
            balance = fdecl.count("(") - fdecl.count(")")
            assert balance >= 0
            if balance > 0:
                continue
            rst_decl = self.parser.parse_func_decl_no_wrap(fdecl)
            fname = rst_decl[0]
            hdr_decls = self.fmap.get(fname, [])
            if not hdr_decls:
                fname = fname.replace("cv.", "")
                hdr_decls = self.fmap.get(fname, [])
            if not hdr_decls:    
                print "Documented function %s (%s) in %s:%d is not in the headers" % (fdecl, rst_decl[0].replace(".", "::"), docname, lineno)
                continue
            decl_idx = 0
            for hd in hdr_decls:
                if len(hd[3]) != len(rst_decl[3]):
                    decl_idx += 1
                    continue
                idx = 0
                for a in hd[3]:
                    if a[0] != rst_decl[3][idx][0] and a[0].replace("cv::", "") != rst_decl[3][idx][0]:
                        break
                    idx += 1
                if idx == len(hd[3]):
                    break
                decl_idx += 1
            if decl_idx < len(hdr_decls):
                self.fmap[fname] = hdr_decls[:decl_idx] + hdr_decls[decl_idx+1:]
                continue
            print "Documented function %s in %s:%d does not have a match" % (fdecl, docname, lineno)
        df.close()

    def decl2str(self, decl):
        return "%s %s(%s)" % (decl[1], decl[0], ", ".join([a[0] + " " + a[1] for a in decl[3]]))

    def check_module_docs(self, name):
        self.parser = hp.CppHeaderParser()
        decls = []
        self.fmap = {}

        for hname in opencv_hdr_list:
            if hname.startswith("../modules/" + name):
                decls += self.parser.parse(hname, wmode=False)
                
        for d in decls:
            fname = d[0]
            if not fname.startswith("struct") and not fname.startswith("class") and not fname.startswith("const"):                
                dlist = self.fmap.get(fname, [])
                dlist.append(d)
                self.fmap[fname] = dlist
    
        self.missing_docfunc_list = []
    
        doclist = glob.glob("../modules/" + name + "/doc/*.rst")
        for d in doclist:
            self.process_rst(d)
            
        print "\n\n########## The list of undocumented functions: ###########\n\n"
        misscount = 0
        fkeys = sorted(self.fmap.keys())
        for f in fkeys:
            # skip undocumented destructors
            if "~" in f:
                continue
            decls = self.fmap[f]
            fcomps = f.split(".")
            prefix = ""
            wlist_decls = []
            for c in fcomps:
                prefix = (prefix + "." + c).lstrip(".")
                wlist_decls = self.whitelist.get(prefix, [])
                if wlist_decls == "*":
                    break
            if wlist_decls == "*":
                continue
            wlist_decls = [self.decl2str(d) for d in wlist_decls]
                
            for d in decls:
                dstr = self.decl2str(d)
                # special hack for ML: skip old variants of the methods
                if name == "ml" and ("CvMat" in dstr):
                    continue
                if dstr not in wlist_decls:
                    misscount += 1
                    print "%s %s(%s)" % (d[1], d[0].replace(".", "::"), ", ".join([a[0] + " " + a[1] for a in d[3]]))
        print "\n\n\nundocumented functions in %s: %d" % (name, misscount)


p = RSTParser()
for m in opencv_module_list:
    print "\n\n*************************** " + m + " *************************\n"
    p.check_module_docs(m)

    
