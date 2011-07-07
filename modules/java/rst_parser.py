import os, sys, re, string, glob
from string import Template

class DeclarationParser(object):
    def __init__(self, line=None):
        if line is None:
            self.fdecl = ""
            self.lang = ""
            self.balance = 0
            return
        self.lang = self.getLang(line)
        assert self.lang is not None
        self.fdecl = line[line.find("::")+2:].strip()
        self.balance = self.fdecl.count("(") - self.fdecl.count(")")
        assert self.balance >= 0

    def append(self, line):
        self.fdecl += line
        self.balance = self.fdecl.count("(") - self.fdecl.count(")")

    def isready(self):
        return self.balance == 0

    def getLang(self, line):
        if line.startswith(".. ocv:function::"):
            return "C++"
        if line.startswith(".. ocv:cfunction::"):
            return "C"
        if line.startswith(".. ocv:pyfunction::"):
            return "Python2"
        if line.startswith(".. ocv:pyoldfunction::"):
            return "Python1"
        if line.startswith(".. ocv:jfunction::"):
            return "Java"
        return None
    
    def hasDeclaration(self, line):
        return self.getLang(line) is not None

class ParamParser(object):
    def __init__(self, line=None):
        if line is None:
            self.prefix = ""
            self.name = ""
            self.comment = ""
            self.active = False
            return
        offset = line.find(":param")
        assert offset > 0
        self.prefix = line[:offset]
        assert self.prefix==" "*len(self.prefix), ":param definition should be prefixed with spaces"
        line = line[offset + 6:].lstrip()
        name_end = line.find(":")
        assert name_end > 0
        self.name = line[:name_end]
        self.comment = line[name_end+1:].lstrip()
        self.active = True

    def append(self, line):
        assert self.active
        if (self.hasDeclaration(line)):
            self.active = False
        elif line.startswith(self.prefix) or not line:
            self.comment += "\n" + line.lstrip()
        else:
            self.active = False
            
    def hasDeclaration(self, line):
        return line.lstrip().startswith(":param")

class RstParser(object):
    def __init__(self, cpp_parser):
        self.cpp_parser = cpp_parser
        self.definitions = {}
        self.sections_parsed = 0
        self.sections_total = 0
        self.sections_skipped = 0

    def parse(self, module_name, module_path=None):
        if module_path is None:
            module_path = "../" + module_name
        doclist = glob.glob(os.path.join(module_path,"doc/*.rst"))
        for doc in doclist:
            self.parse_rst_file(module_name, doc)
            
    def parse_section_safe(self, module_name, section_name, file_name, lineno, lines):
        try:
            self.parse_section(module_name, section_name, file_name, lineno, lines)
        except AssertionError, args:
            print "RST parser error: assertion in \"%s\"  File: %s (line %s)" % (section_name, file_name, lineno)
            print "    Details: %s" % args

    def parse_section(self, module_name, section_name, file_name, lineno, lines):
        self.sections_total += 1
        # skip sections having whitespace in name
        if section_name.find(" ") >= 0 and section_name.find("::operator") < 0:
            print "SKIPPED: \"%s\" File: %s (line %s)" % (section_name, file_name, lineno)
            self.sections_skipped += 1
            return

        func = {}
        func["name"] = section_name
        func["file"] = file_name
        func["line"] = lineno
        func["module"] = module_name

        # parse section name
        section_name = self.parse_namespace(func, section_name)
        class_separator_idx = section_name.find("::")
        if class_separator_idx > 0:
            func["class"] = section_name[:class_separator_idx]
            func["method"] = section_name[class_separator_idx+2:]
        else:
            func["method"] = section_name

        skip_code_lines = False
        expected_brief = True
        fdecl = DeclarationParser()
        pdecl = ParamParser()

        for l in lines:
            # read tail of function/method declaration if needed
            if not fdecl.isready():
                fdecl.append(ll)
                if fdecl.isready():
                    self.add_new_fdecl(func, fdecl)
                continue

            # skip lines if line-skipping mode is activated
            if skip_code_lines:
                if not l or l.startswith(" "):
                    continue
                else:
                    skip_code_lines = False

            ll = l.strip()
            if ll == "..": #strange construction...
                continue

            # turn on line-skipping mode for code fragments
            if ll.endswith("::"):
                skip_code_lines = True
                ll = ll[:len(ll)-3]

            if ll.startswith(".. code-block::"):
                skip_code_lines = True
                continue

            # continue param parsing
            if pdecl.active:
                pdecl.append(l)
                if pdecl.active:
                    continue
                else:
                    self.add_new_pdecl(func, pdecl)
                    # do not continue - current line can contain next parameter definition

            # todo: parse structure members; skip them for now
            if ll.startswith(".. ocv:member::"):
                skip_code_lines = True
                continue

            # todo: parse ".. seealso::" sections

            # parse class & struct definitions
            if ll.startswith(".. ocv:class::"):
                func["class"] = ll[ll.find("::")+2:].strip()
                if "method" in func:
                    del func["method"]
                func["isclass"] = True
                expected_brief = True
                continue

            if ll.startswith(".. ocv:struct::"):
                func["class"] = ll[ll.find("::")+2:].strip()
                if "method" in func:
                    del func["method"]
                func["isstruct"] = True
                expected_brief = True
                continue

            # parse function/method definitions
            if fdecl.hasDeclaration(ll):
                fdecl = DeclarationParser(ll)
                if fdecl.isready():
                    self.add_new_fdecl(func, fdecl)
                expected_brief = False
                continue

            # parse parameters
            if pdecl.hasDeclaration(l):
                pdecl = ParamParser(l)
                expected_brief = False
                continue

            # record brief description
            if expected_brief and len(ll) == 0:
                if "brief" in func:
                    expected_brief = False
                continue
            
            if expected_brief:
                func["brief"] = func.get("brief", "") + "\n" + ll
                if skip_code_lines:
                    expected_brief = False #force end brief if code block begins
                continue

            # record other lines as long description
            func["long"] = func.get("long", "") + "\n" + ll
        # endfor l in lines
        
        if fdecl.balance != 0:
            print "RST parser error: invalid parentheses balance in \"%s\" File: %s (line %s)" % (section_name, file_name, lineno)

        # save last parameter if needed
        if pdecl.active:
            self.add_new_pdecl(func, pdecl)

        # add definition to list
        func = self.normalize(func)
        if self.validate(func):
            self.definitions[func["name"]] = func
            self.sections_parsed += 1
            #self.print_info(func)
        elif func:
            self.print_info(func, True)
            pass

    def parse_rst_file(self, module_name, doc):
        doc = os.path.abspath(doc)
        lineno = 0
        whitespace_warnings = 0
        max_whitespace_warnings = 10
      
        lines = []
        flineno = 0
        fname = ""
        prev_line = None

        df = open(doc, "rt")
        for l in df.readlines():
            lineno += 1
            # handle tabs
            if l.find("\t") >= 0:
                whitespace_warnings += 1
                if whitespace_warnings <= max_whitespace_warnings:
                    print "RST parser warning: tab symbol instead of space is used at file %s (line %s)" % (doc, lineno)
                l = l.replace("\t", "    ")
                
            # handle first line
            if prev_line == None:
                prev_line = l.rstrip()
                continue

            ll = l.rstrip()
            if len(prev_line) > 0 and len(ll) >= len(prev_line) and ll == "-" * len(ll):
                # new function candidate
                if len(lines) > 1:
                    self.parse_section_safe(module_name, fname, doc, flineno, lines[:len(lines)-1])
                lines = []
                flineno = lineno-1
                fname = prev_line.strip()
            elif flineno > 0:
                lines.append(ll)               
            prev_line = ll
        df.close()

        # don't forget about the last function section in file!!!
        if len(lines) > 1:
            self.parse_section_safe(module_name, fname, doc, flineno, lines)

    def parse_namespace(self, func, section_name):
        known_namespaces = ["cv", "gpu", "flann"]
        l = section_name.strip()
        for namespace in known_namespaces:
            if l.startswith(namespace + "::"):
                func["namespace"] = namespace
                return l[len(namespace)+2:]
        return section_name

    def add_new_fdecl(self, func, decl):
        decls =  func.get("decls",[])
        if (decl.lang == "C++" or decl.lang == "C"):
            rst_decl = self.cpp_parser.parse_func_decl_no_wrap(decl.fdecl)
            decls.append( (decl.lang, decl.fdecl, rst_decl) )
        else:
            decls.append( (decl.lang, decl.fdecl) )
        func["decls"] = decls

    def add_new_pdecl(self, func, decl):
        params =  func.get("params",{})
        if decl.name in params:
            print "RST parser error: redefinition of parameter \"%s\" in \"%s\" File: %s (line %s)" \
                 % (decl.name, func["name"], func["file"], func["line"])
        else:
            params[decl.name] = decl.comment
            func["params"] = params

    def print_info(self, func, skipped=False):
        print
        if skipped:
            print "SKIPPED DEFINITION:"
        print "name:      %s" % (func.get("name","~empty~"))
        print "file:      %s (line %s)" % (func.get("file","~empty~"), func.get("line","~empty~"))
        print "is class:  %s" % func.get("isclass",False)
        print "is struct: %s" % func.get("isstruct",False)
        print "module:    %s" % func.get("module","~unknown~")
        print "namespace: %s" % func.get("namespace", "~empty~")
        print "class:     %s" % (func.get("class","~empty~"))
        print "method:    %s" % (func.get("method","~empty~"))
        print "brief:     %s" % (func.get("brief","~empty~"))
        if "decls" in func:
            print "declarations:"
            for d in func["decls"]:
               print "     %7s: %s" % (d[0], re.sub(r"[ ]+", " ", d[1]))
        if "params" in func:
            print "parameters:"
            for name, comment in func["params"].items():
                print "%23s:   %s" % (name, comment)
        if not skipped:
            print "long:      %s" % (func.get("long","~empty~"))
        print

    def validate(self, func):
        if func.get("decls",None) is None:
             if not func.get("isclass",False) and not func.get("isstruct",False):
                 return False
        if func["name"] in self.definitions:
             print "RST parser error: \"%s\" from file: %s (line %s) is already documented in file: %s (line %s)" \
                 % (func["name"], func["file"], func["line"], self.definitions[func["name"]]["file"], self.definitions[func["name"]]["line"])
             return False
        return self.validateParams(func)

    def validateParams(self, func):
        documentedParams = func.get("params",{}).keys()
        params = []
       	
        for decl in func.get("decls", []):
            if len(decl) > 2:
                args = decl[2][3] # decl[2] -> [ funcname, return_ctype, [modifiers], [args] ]
                for arg in args:
                    # arg -> [ ctype, name, def val, [mod], argno ]
                    if arg[0] != "...":
                        params.append(arg[1])
        params = list(set(params))#unique

        # 1. all params are documented
        for p in params:
            if p not in documentedParams:
                print "RST parser warning: parameter \"%s\" of \"%s\" is undocumented. File: %s (line %s)" % (p, func["name"], func["file"], func["line"])

        # 2. only real params are documented
        for p in documentedParams:
            if p not in params:
                print "RST parser warning: unexisting parameter \"%s\" of \"%s\" is documented. File: %s (line %s)" % (p, func["name"], func["file"], func["line"])
        return True

    def normalize(self, func):
        if not func:
            return func
        func["name"] = self.normalizeText(func["name"])
        if "method" in func:
            func["method"] = self.normalizeText(func["method"])
        if "class" in func:
            func["class"] = self.normalizeText(func["class"])
        if "brief" in func:
            func["brief"] = self.normalizeText(func.get("brief",None))
            if not func["brief"]:
                del func["brief"]
        if "long" in func:
            func["long"] = self.normalizeText(func.get("long",None))
            if not func["long"]:
                del func["long"]
        if "decls" in func:
            func["decls"].sort()
        if "params" in func:
            params = {}
            for name, comment in func["params"].items():
                cmt = self.normalizeText(comment)
                if cmt:
                    params[name] = cmt
            func["params"] = params

        # special case for old C functions - section name should omit "cv" prefix
        if not func.get("isclass",False) and not func.get("isstruct",False):
            self.fixOldCFunctionName(func)
        return func

    def fixOldCFunctionName(self, func):
        if not "decls" in func: 
            return
        fname = None
        for decl in func["decls"]:
            if decl[0] != "C" and decl[0] != "Python1":
                return
            if decl[0] == "C":
                fname = decl[2][0]
        if fname is None:
            return

        fname = fname.replace(".", "::")
        if fname.startswith("cv::cv"):
            if fname[6:] == func.get("name", ""):
                func["name"] = fname[4:]
                func["method"] = fname[4:]
            else:
                print "RST parser warning: invalid definition of old C function \"%s\" - section name is \"%s\" instead of \"%s\". File: %s (line %s)" % (fname, func["name"], fname[6:], func["file"], func["line"])
                #self.print_info(func)

    def normalizeText(self, s):
        if s is None:
            return s
        # normalize line endings
        s = re.sub(r"\r\n", "\n", s)
        # remove tailing ::
        s = re.sub(r"::$", "\n", s)
        # remove extra line breaks before/after _ or ,
        s = re.sub(r"\n[ ]*([_,])\n", r"\1", s)
        # remove extra line breaks after `
        #s = re.sub(r"`\n", "` ", s)
        # remove extra line breaks before *
        s = re.sub(r"\n\n\*", "\n\*", s)
        # remove extra line breaks before #.
        s = re.sub(r"\n\n#\.", "\n#.", s)
        # remove extra line breaks after #.
        s = re.sub(r"\n#\.\n", "\n#. ", s)
        # remove extra line breaks before `
        s = re.sub(r"\n[ ]*`", " `", s)
        # remove trailing whitespaces
        s = re.sub(r"[ ]+$", "", s)
        # remove whitespace before .
        s = re.sub(r"[ ]+\.", "\.", s)
        # remove .. for references
        s = re.sub(r"\.\. \[", "[", s)
        # unescape
        s = re.sub(r"\\(.)", "\\1", s)
        # compress whitespace
        s = re.sub(r"[ ]+", " ", s)

        s = s.replace("**", "")
        s = s.replace("``", "\"")
        s = s.replace("`", "\"")
        s = s.replace("\"\"", "\"")
        s = s.replace(":ocv:cfunc:","")
        s = s.replace(":math:", "")
        s = s.replace(":ocv:class:", "")
        s = s.replace(":ocv:func:", "")
        s = s.replace("]_", "]")
        s = s.strip()
        return s

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print "Usage:\n", os.path.basename(sys.argv[0]), " <module path>"
        exit(0)

    rst_parser_dir  = os.path.dirname(os.path.abspath(sys.argv[0]))
    hdr_parser_path = os.path.join(rst_parser_dir, "../python/src2")

    sys.path.append(hdr_parser_path)
    import hdr_parser

    module = sys.argv[1]

    if module != "all" and not os.path.isdir(os.path.join(rst_parser_dir, "../" + module)):
        print "Module \"" + module + "\" could not be found."
        exit(1)

    parser = RstParser(hdr_parser.CppHeaderParser())
    
    if module == "all":
        for m in ["core", "flann", "imgproc", "ml", "highgui", "video", "features2d", "calib3d", "objdetect", "legacy", "contrib", "gpu", "androidcamera", "haartraining", "java", "ocl", "python", "stitching", "traincascade", "ts"]:
            parser.parse(m, os.path.join(rst_parser_dir, "../" + m))
    else:
        parser.parse(module, os.path.join(rst_parser_dir, "../" + module))

    # summary
    print
    print "RST Parser Summary:"
    print "  Total sections:   %s" % parser.sections_total
    print "  Skipped sections: %s" % parser.sections_skipped
    print "  Parsed  sections: %s" % parser.sections_parsed
    print "  Invalid sections: %s" % (parser.sections_total - parser.sections_parsed - parser.sections_skipped)

    # statistic by language
    stat = {}
    classes = 0
    structs = 0
    for name, d in parser.definitions.items():
       if d.get("isclass", False):
           classes += 1
       elif d.get("isstruct", False):
           structs += 1
       else:
           for decl in d.get("decls",[]):
               stat[decl[0]] = stat.get(decl[0],0) + 1

    print
    print "  classes documented:           %s" % classes
    print "  structs documented:           %s" % structs
    for lang in sorted(stat.items()):
        print "  %7s functions documented: %s" % lang


