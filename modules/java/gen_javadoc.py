import os, sys, re, string, glob

javadoc_marker = "//javadoc:"

def parceJavadocMarker(line):
    assert line.lstrip().startswith(javadoc_marker)
    offset = line[:line.find(javadoc_marker)]
    line = line.strip()[len(javadoc_marker):]
    args_start = line.rfind("(")
    args_end = line.rfind(")")
    assert args_start * args_end > 0
    if args_start >= 0:
        assert args_start < args_end
        return (line[:args_start].strip(), offset,  filter(None, list(arg.strip() for arg in line[args_start+1:args_end].split(","))))
    return (line, offset, [])

def document(infile, outfile, decls):
    inf = open(infile, "rt")
    outf = open(outfile, "wt")
    try:
        for l in inf.readlines():
            if l.lstrip().startswith(javadoc_marker):
                marker = parceJavadocMarker(l)
                decl = decls.get(marker[0],None)
                if decl:
                    for line in makeJavadoc(decl, decls, marker[2]).split("\n"):
                        outf.write(marker[1] + line + "\n")
                else:
                    print "Error: could not find documentation for %s" % l.lstrip()[len(javadoc_marker):-1]
            else:
                outf.write(l.replace("\t", "    ").rstrip()+"\n")
    except:
        inf.close()
        outf.close()
        os.remove(outfile)
        raise
    else:
        inf.close()
        outf.close()

def ReformatForJavadoc(s):
    out = ""
    for term in s.split("\n"):
        if term.startswith("*") or term.startswith("#."):
            term = "  " + term
        if not term:
            out += " *\n"
        else:
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
                out += " * " + term[pos_start:pos_end+1].rstrip() + "\n"
                pos_start = pos_end + 1
                pos_end = min(pos_start + 77, len(term)-1)
    return out

def getJavaName(decl):
    name = "org.opencv."
    name += decl["module"]
    if "class" in decl:
        name += "." + decl["class"]
    else:
        name += "." + decl["module"].capitalize()
    if "method" in decl:
        name += "." + decl["method"]
    return name

def getDocURL(decl):
    url = "http://opencv.itseez.com/modules/"
    url += decl["module"]
    url += "/doc/"
    url += os.path.basename(decl["file"]).replace(".rst",".html")
    url += "#" + decl["name"].replace("::","-").replace("()","").replace("=","").strip().rstrip("_").replace(" ","-").replace("_","-").lower()
    return url

def makeJavadoc(decl, decls, args = None):
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
        doc += prefix + ReformatForJavadoc(decl["brief"])
        prefix = " *\n"
    elif "long" not in decl:
        print "Warning: no description for " + decl_type + " \"%s\" File: %s (line %s)" % (func["name"], func["file"], func["line"])
        doc += prefix + ReformatForJavadoc("This " + decl_type + " is undocumented")
        prefix = " *\n"
    
    # long goes after brief
    if "long" in decl:
        doc += prefix  + ReformatForJavadoc(decl["long"])
        prefix = " *\n"

    # @param tags
    if args and (decl_type == "method" or decl_type == "function"):
        documented_params = decl.get("params",{})
        for arg in args:
            arg_doc = documented_params.get(arg, None)
            if not arg_doc:
                arg_doc = "a " + arg
                print "Warning: parameter \"%s\" of \"%s\" is undocumented. File: %s (line %s)" % (arg, decl["name"], decl["file"], decl["line"])
            doc += prefix + ReformatForJavadoc("@param " + arg + " " + arg_doc)
            prefix = ""
        prefix = " *\n"

    # @see tags
    # always link to documentation
    doc += prefix + " * @see <a href=\"" + getDocURL(decl) + "\">" + getJavaName(decl) + "</a>\n"
    prefix = ""
    # other links
    if "seealso" in decl:
        for see in decl["seealso"]:
            seedecl = decls.get(see,None)
            if seedecl:
                doc += prefix + " * @see " + getJavaName(seedecl) + "\n"
            else:
                doc += prefix + " * @see " + see.replace("::",".") + "\n"
    prefix = " *\n"

    #doc += prefix + " * File: " + decl["file"] + " (line " + str(decl["line"]) + ")\n"

    return (doc + " */").replace("::",".")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage:\n", os.path.basename(sys.argv[0]), " <input dir1> [<input dir2> [...]]"
        exit(0)
   
    selfpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    hdr_parser_path = os.path.join(selfpath, "../python/src2")
    
    sys.path.append(selfpath)
    sys.path.append(hdr_parser_path)
    import hdr_parser
    import rst_parser

    parser = rst_parser.RstParser(hdr_parser.CppHeaderParser())
    
    print "Parsing documentation..."
    for m in ["core", "flann", "imgproc", "ml", "highgui", "video", "features2d", "calib3d", "objdetect", "legacy", "contrib", "gpu", "androidcamera", "haartraining", "java", "python", "stitching", "traincascade", "ts"]:
        parser.parse(m, os.path.join(selfpath, "../" + m))
        
    parser.printSummary()

    for i in range(1, len(sys.argv)):
        folder = os.path.abspath(sys.argv[i])
        for jfile in [f for f in glob.glob(os.path.join(folder,"*.java")) if not f.endswith("-jdoc.java")]:
            outfile = os.path.abspath(os.path.basename(jfile).replace(".java", "-jdoc.java"))
            document(jfile, outfile, parser.definitions)
