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
        assert self.prefix==" "*len(self.prefix)
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

    def parse(self, module_path):
        doclist = glob.glob(os.path.join(module_path,"doc/*.rst"))
        for doc in doclist:
            self.parse_rst_file(doc)

    def parse_section(self, section_name, file_name, lineno, lines):
        func = {}
        func["name"] = section_name
        func["file"] = file_name
        func["line"] = lineno

        # parse section name
        class_separator_idx = func["name"].find("::")
        if class_separator_idx > 0:
            func["class"] = func["name"][:class_separator_idx]
            func["method"] = func["name"][class_separator_idx+2:]
        else:
            func["method"] = func["name"]

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
                if not l or l.startswith(" ") or l.startswith("\t"):
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
                    #do not continue - current line can contain next parameter definition

            # todo: parse structure members; skip them for now
            if ll.startswith(".. ocv:member::"):
                skip_code_lines = True
                continue

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

        # save last parameter if needed
        if pdecl.active:
            self.add_new_pdecl(func, pdecl)

        # add definition to list
        func = self.normalize(func)
        if self.validate(func):
            self.definitions[func["name"]] = func
            #self.print_info(func)
        elif func:
            self.print_info(func, True)

    def parse_rst_file(self, doc):
        doc = os.path.abspath(doc)
        lineno = 0
      
        lines = []
        flineno = 0
        fname = ""
        prev_line = None

        df = open(doc, "rt")
        for l in df.readlines():
            lineno += 1
            if prev_line == None:
                prev_line = l.rstrip()
                continue
            ll = l.rstrip()
            if len(prev_line) > 0 and len(ll) >= len(prev_line) and ll == "-" * len(ll):
                #new function candidate
                if len(lines) > 1:
                    self.parse_section(fname, doc, flineno, lines[:len(lines)-1])
                lines = []
                flineno = lineno-1
                fname = prev_line.strip()
            elif flineno > 0:
                lines.append(ll)               
            prev_line = ll
        df.close()

        #don't forget about the last function section in file!!!
        if len(lines) > 1:
            self.parse_section(fname, doc, flineno, lines[:len(lines)])

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
            print "Parser error: parameter \"%s\" for %s is defined multiple times. See %s line %s" \
                 % (decl.name, func["name"], func["file"], func["line"])
        else:
            params[decl.name] = decl.comment
            func["params"] = params

    def print_info(self, func, skipped=False):
        print ""
        if skipped:
            print "SKIPPED DEFINITION:"
        print "name:      %s" % (func.get("name","~empty~"))
        print "file:      %s (line %s)" % (func.get("file","~empty~"), func.get("line","~empty~"))
        print "is class:  %s" % func.get("isclass",False)
        print "is struct: %s" % func.get("isstruct",False)
        print "class:     %s" % (func.get("class","~empty~"))
        print "method:    %s" % (func.get("method","~empty~"))
        print "brief:     %s" % (func.get("brief","~empty~"))
        if "decls" in func:
            print "declarations:"
            for d in func["decls"]:
               print "     %7s: %s" % (d[0], re.sub(r"[ \t]+", " ", d[1]))
        if "params" in func:
            print "parameters:"
            for name, comment in func["params"].items():
                print "%23s:   %s" % (name, comment)
        if not skipped:
            print "long:      %s" % (func.get("long","~empty~"))

    def validate(self, func):
        if func.get("decls",None) is None:
             if not func.get("isclass",False):
                 return False
        if func["name"] in self.definitions:
             print "Parser error: function/class/struct \"%s\" in %s line %s is already documented in %s line %s" \
                 % (func["name"], func["file"], func["line"], self.definitions[func["name"]]["file"], self.definitions[func["name"]]["line"])
             return False
        #todo: validate parameter names
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
        return func

    def normalizeText(self, s):
        if s is None:
            return s
        # normalize line endings
        s = re.sub(r"\r\n", "\n", s)
        # remove tailing ::
        s = re.sub(r"::$", "\n", s)
        # remove extra line breaks before/after _ or ,
        s = re.sub(r"\n[ \t]*([_,])\n", r"\1", s)
        # remove extra line breaks after `
        #s = re.sub(r"`\n", "` ", s)
        # remove extra line breaks before *
        s = re.sub(r"\n\n\*", "\n\*", s)
        # remove extra line breaks before #.
        s = re.sub(r"\n\n#\.", "\n#.", s)
        # remove extra line breaks after #.
        s = re.sub(r"\n#\.\n", "\n#. ", s)
        # remove extra line breaks before `
        s = re.sub(r"\n[ \t]*`", " `", s)
        # remove trailing whitespaces
        s = re.sub(r"[ \t]+$", "", s)
        # remove whitespace before .
        s = re.sub(r"[ \t]+\.", "\.", s)
        # remove .. for references
        s = re.sub(r"\.\. \[", "[", s)
        # unescape
        s = re.sub(r"\\(.)", "\\1", s)
        # compress whitespace
        s = re.sub(r"[ \t]+", " ", s)

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

    if not os.path.isdir(os.path.join(rst_parser_dir, "../" + module)):
        print "Module \"" + module + "\" could not be found."
        exit(1)

    parser = RstParser(hdr_parser.CppHeaderParser())
    parser.parse(os.path.join(rst_parser_dir, "../" + module))

