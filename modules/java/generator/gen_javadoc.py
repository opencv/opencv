#!/usr/bin/env python

import os, sys, re, string, glob
from optparse import OptionParser

# Black list for classes and methods that does not implemented in Java API
# Created to exclude referencies to them in @see tag
JAVADOC_ENTITY_BLACK_LIST = set(["org.opencv.core.Core#abs", \
                                 "org.opencv.core.Core#theRNG", \
                                 "org.opencv.core.Core#extractImageCOI", \
                                 "org.opencv.core.PCA", \
                                 "org.opencv.core.SVD", \
                                 "org.opencv.core.RNG", \
                                 "org.opencv.imgproc.Imgproc#createMorphologyFilter", \
                                 "org.opencv.imgproc.Imgproc#createLinearFilter", \
                                 "org.opencv.imgproc.Imgproc#createSeparableLinearFilter", \
                                 "org.opencv.imgproc.FilterEngine"])

class JavadocGenerator(object):
    def __init__(self, definitions = {}, modules= [], javadoc_marker = "//javadoc:"):
        self.definitions = definitions
        self.javadoc_marker = javadoc_marker
        self.markers_processed = 0
        self.markers_documented = 0
        self.params_documented = 0
        self.params_undocumented = 0
        self.known_modules = modules
        self.verbose = False
        self.show_warnings = True
        self.show_errors = True

    def parceJavadocMarker(self, line):
        assert line.lstrip().startswith(self.javadoc_marker)
        offset = line[:line.find(self.javadoc_marker)]
        line = line.strip()[len(self.javadoc_marker):]
        args_start = line.rfind("(")
        args_end = line.rfind(")")
        assert args_start * args_end > 0
        if args_start >= 0:
            assert args_start < args_end
            name = line[:args_start].strip()
            if name.startswith("java"):
                name = name[4:]
            return (name, offset,  filter(None, list(arg.strip() for arg in line[args_start+1:args_end].split(","))))
        name = line.strip()
        if name.startswith("java"):
            name = name[4:]
        return (name, offset, [])

    def document(self, infile, outfile):
        inf = open(infile, "rt")
        outf = open(outfile, "wt")
        module = os.path.splitext(os.path.basename(infile))[0].split("+")[0]
        if module not in self.known_modules:
            module = "unknown"
        try:
            for l in inf.readlines():
                org = l
                l = l.replace(" ", "").replace("\t", "")#remove all whitespace
                if l.startswith(self.javadoc_marker):
                    marker = self.parceJavadocMarker(l)
                    self.markers_processed += 1
                    decl = self.definitions.get(marker[0],None)
                    if decl:
                        javadoc = self.makeJavadoc(decl, marker[2])
                        if self.verbose:
                            print
                            print "Javadoc for \"%s\" File: %s (line %s)" % (decl["name"], decl["file"], decl["line"])
                            print javadoc
                        for line in javadoc.split("\n"):
                            outf.write(marker[1] + line + "\n")
                        self.markers_documented += 1
                    elif self.show_errors:
                        print >> sys.stderr, "gen_javadoc error: could not find documentation for %s (module: %s)" % (l.lstrip()[len(self.javadoc_marker):-1].strip(), module)
                else:
                    outf.write(org.replace("\t", "    ").rstrip()+"\n")
        except:
            inf.close()
            outf.close()
            os.remove(outfile)
            raise
        else:
            inf.close()
            outf.close()

    def FinishParagraph(self, text):
        return text[:-1] + "</p>\n"

    def ReformatForJavadoc(self, s):
        out = ""
        in_paragraph = False
        in_list = False
        for term in s.split("\n"):
            in_list_item = False
            if term.startswith("*"):
                in_list_item = True
                if in_paragraph:
                    out = self.FinishParagraph(out)
                    in_paragraph = False
                if not in_list:
                    out += " * <ul>\n"
                    in_list = True
                term = "  <li>" + term[1:]

            if term.startswith("#."):
                in_list_item = True
                if in_paragraph:
                    out = self.FinishParagraph(out)
                    in_paragraph = False
                if not in_list:
                    out += " * <ul>\n"
                    in_list = True
                term = "  <li>" + term[2:]

            if not term:
                if in_paragraph:
                    out = self.FinishParagraph(out)
                    in_paragraph = False
                out += " *\n"
            else:
                if in_list and not in_list_item:
                    in_list = False
                    if out.endswith(" *\n"):
                        out = out[:-3] + " * </ul>\n *\n"
                    else:
                        out += " * </ul>\n"
                pos_start = 0
                pos_end = min(77, len(term)-1)
                while pos_start < pos_end:
                    if pos_end - pos_start == 77:
                        while pos_end >= pos_start+60:
                            if not term[pos_end].isspace():
                                pos_end -= 1
                            else:
                                break
                        if pos_end < pos_start+60:
                            pos_end = min(pos_start + 77, len(term)-1)
                            while pos_end < len(term):
                                if not term[pos_end].isspace():
                                    pos_end += 1
                                else:
                                    break
                    if in_paragraph or term.startswith("@") or in_list_item:
                        out += " * "
                    else:
                        in_paragraph = True
                        out += " * <p>"
                    out += term[pos_start:pos_end+1].rstrip() + "\n"
                    pos_start = pos_end + 1
                    pos_end = min(pos_start + 77, len(term)-1)

        if in_paragraph:
            out = self.FinishParagraph(out)
        if in_list:
            out += " * </ul>\n"
        return out

    def getJavaName(self, decl, methodSeparator = "."):
        name = "org.opencv."
        name += decl["module"]
        if "class" in decl:
            name += "." + decl["class"]
        else:
            name += "." + decl["module"].capitalize()
        if "method" in decl:
            name += methodSeparator + decl["method"]
        return name

    def getDocURL(self, decl):
        url = "http://docs.opencv.org/modules/"
        url += decl["module"]
        url += "/doc/"
        url += os.path.basename(decl["file"]).replace(".rst",".html")
        url += "#" + decl["name"].replace("::","-").replace("()","").replace("=","").strip().rstrip("_").replace(" ","-").replace("_","-").lower()
        return url

    def makeJavadoc(self, decl, args = None):
        doc = ""
        prefix = "/**\n"

        if decl.get("isclass", False):
            decl_type = "class"
        elif decl.get("isstruct", False):
            decl_type = "struct"
        elif "class" in decl:
            decl_type = "method"
        else:
            decl_type = "function"

        # brief goes first
        if "brief" in decl:
            doc += prefix + self.ReformatForJavadoc(decl["brief"])
            prefix = " *\n"
        elif "long" not in decl:
            if self.show_warnings:
                print >> sys.stderr, "gen_javadoc warning: no description for " + decl_type + " \"%s\" File: %s (line %s)" % (func["name"], func["file"], func["line"])
            doc += prefix + self.ReformatForJavadoc("This " + decl_type + " is undocumented")
            prefix = " *\n"

        # long goes after brief
        if "long" in decl:
            doc += prefix  + self.ReformatForJavadoc(decl["long"])
            prefix = " *\n"

        # @param tags
        if args and (decl_type == "method" or decl_type == "function"):
            documented_params = decl.get("params",{})
            for arg in args:
                arg_doc = documented_params.get(arg, None)
                if not arg_doc:
                    arg_doc = "a " + arg
                    if self.show_warnings:
                        print >> sys.stderr, "gen_javadoc warning: parameter \"%s\" of \"%s\" is undocumented. File: %s (line %s)" % (arg, decl["name"], decl["file"], decl["line"])
                    self.params_undocumented += 1
                else:
                    self.params_documented += 1
                doc += prefix + self.ReformatForJavadoc("@param " + arg + " " + arg_doc)
                prefix = ""
            prefix = " *\n"

        # @see tags
        # always link to documentation
        doc += prefix + " * @see <a href=\"" + self.getDocURL(decl) + "\">" + self.getJavaName(decl) + "</a>\n"
        prefix = ""
        # other links
        if "seealso" in decl:
            for see in decl["seealso"]:
                seedecl = self.definitions.get(see,None)
                if seedecl:
                    javadoc_name = self.getJavaName(seedecl, "#")
                    if (javadoc_name not in JAVADOC_ENTITY_BLACK_LIST):
                        doc += prefix + " * @see " + javadoc_name + "\n"
        prefix = " *\n"

        #doc += prefix + " * File: " + decl["file"] + " (line " + str(decl["line"]) + ")\n"

        return (doc + " */").replace("::",".")

    def printSummary(self):
        print "Javadoc Generator Summary:"
        print "  Total markers:        %s" % self.markers_processed
        print "  Undocumented markers: %s" % (self.markers_processed - self.markers_documented)
        print "  Generated comments:   %s" % self.markers_documented

        print
        print "  Documented params:    %s" % self.params_documented
        print "  Undocumented params:  %s" % self.params_undocumented
        print

if __name__ == "__main__":

    selfpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    hdr_parser_path = os.path.join(selfpath, "../../python/src2")

    sys.path.append(selfpath)
    sys.path.append(hdr_parser_path)
    import hdr_parser
    import rst_parser

    parser = OptionParser()
    parser.add_option("-v", "--verbose", dest="verbose", help="Print verbose log to stdout", action="store_true", default=False)
    parser.add_option("", "--no-warnings", dest="warnings", help="Hide warning messages", action="store_false", default=True)
    parser.add_option("", "--no-errors", dest="errors", help="Hide error messages", action="store_false", default=True)
    parser.add_option("", "--modules", dest="modules", help="comma-separated list of modules to generate comments", metavar="MODS", default=",".join(rst_parser.allmodules))

    (options, args) = parser.parse_args(sys.argv)
    options.modules = options.modules.split(",")

    if len(args) < 2 or len(options.modules) < 1:
        parser.print_help()
        exit(0)

    parser = rst_parser.RstParser(hdr_parser.CppHeaderParser())
    for m in options.modules:
        parser.parse(m, os.path.join(selfpath, "../../" + m))

    parser.printSummary()

    generator = JavadocGenerator(parser.definitions, options.modules)
    generator.verbose = options.verbose
    generator.show_warnings = options.warnings
    generator.show_errors = options.errors

    for path in args:
        folder = os.path.abspath(path)
        for jfile in [f for f in glob.glob(os.path.join(folder,"*.java")) if not f.endswith("-jdoc.java")]:
            outfile = os.path.abspath(os.path.basename(jfile).replace(".java", "-jdoc.java"))
            generator.document(jfile, outfile)

    generator.printSummary()
