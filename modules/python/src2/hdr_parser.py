import os, sys, re, string

# the list only for debugging. The real list, used in the real OpenCV build, is specified in CMakeLists.txt
opencv_hdr_list = [
"../../core/include/opencv2/core/core.hpp",
"../../flann/include/opencv2/flann/miniflann.hpp",
"../../ml/include/opencv2/ml/ml.hpp",
"../../imgproc/include/opencv2/imgproc/imgproc.hpp",
"../../calib3d/include/opencv2/calib3d/calib3d.hpp",
"../../features2d/include/opencv2/features2d/features2d.hpp",
"../../video/include/opencv2/video/tracking.hpp",
"../../video/include/opencv2/video/background_segm.hpp",
"../../objdetect/include/opencv2/objdetect/objdetect.hpp",
"../../highgui/include/opencv2/highgui/highgui.hpp"
]

"""
Each declaration is [funcname, return_value_type /* in C, not in Python */, <list_of_modifiers>, <list_of_arguments>],
where each element of <list_of_arguments> is 4-element list itself:
[argtype, argname, default_value /* or "" if none */, <list_of_modifiers>]
where the list of modifiers is yet another nested list of strings
   (currently recognized are "/O" for output argument, "/S" for static (i.e. class) methods
   and "/A value" for the plain C arrays with counters)
"""

class CppHeaderParser(object):

    def __init__(self):
        self.BLOCK_TYPE = 0
        self.BLOCK_NAME = 1
        self.PROCESS_FLAG = 2
        self.PUBLIC_SECTION = 3
        self.CLASS_DECL = 4

    def batch_replace(self, s, pairs):
        for before, after in pairs:
            s = s.replace(before, after)
        return s

    def get_macro_arg(self, arg_str, npos):
        npos2 = npos3 = arg_str.find("(", npos)
        if npos2 < 0:
            print "Error: no arguments for the macro at %d" % (self.lineno,)
            sys.exit(-1)
        balance = 1
        while 1:
            t, npos3 = self.find_next_token(arg_str, ['(', ')'], npos3+1)
            if npos3 < 0:
                print "Error: no matching ')' in the macro call at %d" % (self.lineno,)
                sys.exit(-1)
            if t == '(':
                balance += 1
            if t == ')':
                balance -= 1
                if balance == 0:
                    break

        return arg_str[npos2+1:npos3].strip(), npos3

    def parse_arg(self, arg_str, argno):
        """
        Parses <arg_type> [arg_name]
        Returns arg_type, arg_name, modlist, argno, where
        modlist is the list of wrapper-related modifiers (such as "output argument", "has counter", ...)
        and argno is the new index of an anonymous argument.
        That is, if no arg_str is just an argument type without argument name, the argument name is set to
        "arg" + str(argno), and then argno is incremented.
        """
        modlist = []

        # pass 0: extracts the modifiers
        if "CV_OUT" in arg_str:
            modlist.append("/O")
            arg_str = arg_str.replace("CV_OUT", "")

        if "CV_IN_OUT" in arg_str:
            modlist.append("/IO")
            arg_str = arg_str.replace("CV_IN_OUT", "")

        isarray = False
        npos = arg_str.find("CV_CARRAY")
        if npos >= 0:
            isarray = True
            macro_arg, npos3 = self.get_macro_arg(arg_str, npos)

            modlist.append("/A " + macro_arg)
            arg_str = arg_str[:npos] + arg_str[npos3+1:]

        npos = arg_str.find("CV_CUSTOM_CARRAY")
        if npos >= 0:
            isarray = True
            macro_arg, npos3 = self.get_macro_arg(arg_str, npos)

            modlist.append("/CA " + macro_arg)
            arg_str = arg_str[:npos] + arg_str[npos3+1:]

        arg_str = arg_str.strip()
        word_start = 0
        word_list = []
        npos = -1

        #print self.lineno, ":\t", arg_str

        # pass 1: split argument type into tokens
        while 1:
            npos += 1
            t, npos = self.find_next_token(arg_str, [" ", "&", "*", "<", ">", ","], npos)
            w = arg_str[word_start:npos].strip()
            if w == "operator":
                word_list.append("operator " + arg_str[npos:].strip())
                break
            if w not in ["", "const"]:
                word_list.append(w)
            if t not in ["", " ", "&"]:
                word_list.append(t)
            if not t:
                break
            word_start = npos+1
            npos = word_start - 1

        arg_type = ""
        arg_name = ""
        angle_stack = []

        #print self.lineno, ":\t", word_list

        # pass 2: decrypt the list
        wi = -1
        prev_w = ""
        for w in word_list:
            wi += 1
            if w == "*":
                if prev_w == "char" and not isarray:
                    arg_type = arg_type[:-len("char")] + "c_string"
                else:
                    arg_type += w
                continue
            elif w == "<":
                arg_type += "_"
                angle_stack.append(0)
            elif w == "," or w == '>':
                if not angle_stack:
                    print "Error at %d: argument contains ',' or '>' not within template arguments" % (self.lineno,)
                    sys.exit(-1)
                if w == ",":
                    arg_type += "_and_"
                elif w == ">":
                    if angle_stack[0] == 0:
                        print "Error at %s:%d: template has no arguments" % (self.hname, self.lineno)
                        sys.exit(-1)
                    if angle_stack[0] > 1:
                        arg_type += "_end_"
                    angle_stack[-1:] = []
            elif angle_stack:
                arg_type += w
                angle_stack[-1] += 1
            elif arg_type and arg_type != "~":
                arg_name = " ".join(word_list[wi:])
                break
            else:
                arg_type += w
            prev_w = w

        counter_str = ""
        add_star = False
        if ("[" in arg_name) and not ("operator" in arg_str):
            #print arg_str
            p1 = arg_name.find("[")
            p2 = arg_name.find("]",p1+1)
            if p2 < 0:
                print "Error at %d: no closing ]" % (self.lineno,)
                sys.exit(-1)
            counter_str = arg_name[p1+1:p2].strip()
            if counter_str == "":
                counter_str = "?"
            if not isarray:
                modlist.append("/A " + counter_str.strip())
            arg_name = arg_name[:p1]
            add_star = True

        if not arg_name:
            if arg_type.startswith("operator"):
                arg_type, arg_name = "", arg_type
            else:
                arg_name = "arg" + str(argno)
                argno += 1

        while arg_type.endswith("_end_"):
            arg_type = arg_type[:-len("_end_")]

        if add_star:
            arg_type += "*"

        arg_type = self.batch_replace(arg_type, [("std::", ""), ("cv::", ""), ("::", "_")])

        return arg_type, arg_name, modlist, argno

    def parse_enum(self, decl_str):
        l = decl_str
        ll = l.split(",")
        prev_val = ""
        prev_val_delta = -1
        decl = []
        for pair in ll:
            pv = pair.split("=")
            if len(pv) == 1:
                prev_val_delta += 1
                val = ""
                if prev_val:
                    val = prev_val + "+"
                val += str(prev_val_delta)
            else:
                prev_val_delta = 0
                prev_val = val = pv[1].strip()
            decl.append(["const " + self.get_dotted_name(pv[0].strip()), val, [], []])
        return decl

    def parse_class_decl(self, decl_str):
        """
        Parses class/struct declaration start in the form:
           {class|struct} [CV_EXPORTS] <class_name> [: public <base_class1> [, ...]]
        Returns class_name1, <list of base_classes>
        """
        l = decl_str
        modlist = []
        if "CV_EXPORTS_W_MAP" in l:
            l = l.replace("CV_EXPORTS_W_MAP", "")
            modlist.append("/Map")
        if "CV_EXPORTS_W_SIMPLE" in l:
            l = l.replace("CV_EXPORTS_W_SIMPLE", "")
            modlist.append("/Simple")
        npos = l.find("CV_EXPORTS_AS")
        if npos >= 0:
            macro_arg, npos3 = self.get_macro_arg(l, npos)
            modlist.append("=" + macro_arg)
            l = l[:npos] + l[npos3+1:]

        l = self.batch_replace(l, [("CV_EXPORTS_W", ""), ("CV_EXPORTS", ""), ("public virtual ", " "), ("public ", " "), ("::", ".")]).strip()
        ll = re.split(r'\s*[,:]?\s*', l)
        ll = [le for le in ll if le]
        classname = ll[1]
        bases = ll[2:]
        return classname, bases, modlist

    def parse_func_decl_no_wrap(self, decl_str, static_method = False):
        decl_str = (decl_str or "").strip()
        virtual_method = False
        explicit_method = False
        if decl_str.startswith("explicit"):
            decl_str = decl_str[len("explicit"):].lstrip()
            explicit_method = True
        if decl_str.startswith("virtual"):
            decl_str = decl_str[len("virtual"):].lstrip()
            virtual_method = True
        if decl_str.startswith("static"):
            decl_str = decl_str[len("static"):].lstrip()
            static_method = True

        fdecl = decl_str.replace("CV_OUT", "").replace("CV_IN_OUT", "")
        fdecl = fdecl.strip().replace("\t", " ")
        while "  " in fdecl:
            fdecl = fdecl.replace("  ", " ")
        fname = fdecl[:fdecl.find("(")].strip()
        fnpos = fname.rfind(" ")
        if fnpos < 0:
            fnpos = 0
        fname = fname[fnpos:].strip()
        rettype = fdecl[:fnpos].strip()

        if rettype.endswith("operator"):
            fname = ("operator " + fname).strip()
            rettype = rettype[:rettype.rfind("operator")].strip()
            if rettype.endswith("::"):
                rpos = rettype.rfind(" ")
                if rpos >= 0:
                    fname = rettype[rpos+1:].strip() + fname
                    rettype = rettype[:rpos].strip()
                else:
                    fname = rettype + fname
                    rettype = ""

        apos = fdecl.find("(")
        if fname.endswith("operator"):
            fname += "()"
            apos = fdecl.find("(", apos+1)

        fname = "cv." + fname.replace("::", ".")
        decl = [fname, rettype, [], []]

        # inline constructor implementation
        implmatch = re.match(r"(\(.*?\))\s*:\s*(\w+\(.*?\),?\s*)+", fdecl[apos:])
        if bool(implmatch):
            fdecl = fdecl[:apos] + implmatch.group(1)

        args0str = fdecl[apos+1:fdecl.rfind(")")].strip()

        if args0str != "" and args0str != "void":
            args0str = re.sub(r"\([^)]*\)", lambda m: m.group(0).replace(',', "@comma@"), args0str)
            args0 = args0str.split(",")

            args = []
            narg = ""
            for arg in args0:
                narg += arg.strip()
                balance_paren = narg.count("(") - narg.count(")")
                balance_angle = narg.count("<") - narg.count(">")
                if balance_paren == 0 and balance_angle == 0:
                    args.append(narg.strip())
                    narg = ""

            for arg in args:
                dfpos = arg.find("=")
                defval = ""
                if dfpos >= 0:
                    defval = arg[dfpos+1:].strip()
                else:
                    dfpos = arg.find("CV_DEFAULT")
                    if dfpos >= 0:
                        defval, pos3 = self.get_macro_arg(arg, dfpos)
                    else:
                        dfpos = arg.find("CV_WRAP_DEFAULT")
                        if dfpos >= 0:
                            defval, pos3 = self.get_macro_arg(arg, dfpos)
                if dfpos >= 0:
                    defval = defval.replace("@comma@", ",")
                    arg = arg[:dfpos].strip()
                pos = len(arg)-1
                while pos >= 0 and (arg[pos] in "_[]" or arg[pos].isalpha() or arg[pos].isdigit()):
                    pos -= 1
                if pos >= 0:
                    aname = arg[pos+1:].strip()
                    atype = arg[:pos+1].strip()
                    if aname.endswith("&") or aname.endswith("*") or (aname in ["int", "string", "Mat"]):
                        atype = (atype + " " + aname).strip()
                        aname = "param"
                else:
                    atype = arg
                    aname = "param"
                decl[3].append([atype, aname, defval, []])

        if static_method:
            decl[2].append("/S")
        if virtual_method:
            decl[2].append("/V")
        if explicit_method:
            decl[2].append("/E")
        if bool(re.match(r".*\)\s*(const)?\s*=\s*0", decl_str)):
            decl[2].append("/A")
        if bool(re.match(r".*\)\s*const(\s*=\s*0)?", decl_str)):
            decl[2].append("/C")
        if "virtual" in decl_str:
            print decl_str
        return decl

    def parse_func_decl(self, decl_str):
        """
        Parses the function or method declaration in the form:
        [([CV_EXPORTS] <rettype>) | CVAPI(rettype)]
            [~]<function_name>
            (<arg_type1> <arg_name1>[=<default_value1>] [, <arg_type2> <arg_name2>[=<default_value2>] ...])
            [const] {; | <function_body>}

        Returns the function declaration entry:
        [<func name>, <return value C-type>, <list of modifiers>, <list of arguments>] (see above)
        """

        if self.wrap_mode:
            if not (("CV_EXPORTS_AS" in decl_str) or ("CV_EXPORTS_W" in decl_str) or \
                ("CV_WRAP" in decl_str) or ("CV_WRAP_AS" in decl_str)):
                return []

        # ignore old API in the documentation check (for now)
        if "CVAPI(" in decl_str and self.wrap_mode:
            return []

        top = self.block_stack[-1]
        func_modlist = []

        npos = decl_str.find("CV_EXPORTS_AS")
        if npos >= 0:
            arg, npos3 = self.get_macro_arg(decl_str, npos)
            func_modlist.append("="+arg)
            decl_str = decl_str[:npos] + decl_str[npos3+1:]
        npos = decl_str.find("CV_WRAP_AS")
        if npos >= 0:
            arg, npos3 = self.get_macro_arg(decl_str, npos)
            func_modlist.append("="+arg)
            decl_str = decl_str[:npos] + decl_str[npos3+1:]

        # filter off some common prefixes, which are meaningless for Python wrappers.
        # note that we do not strip "static" prefix, which does matter;
        # it means class methods, not instance methods
        decl_str = self.batch_replace(decl_str, [("virtual", ""), ("static inline", ""), ("inline", ""),\
            ("CV_EXPORTS_W", ""), ("CV_EXPORTS", ""), ("CV_CDECL", ""), ("CV_WRAP ", " "), ("static CV_INLINE", ""), ("CV_INLINE", "")]).strip()

        static_method = False
        context = top[0]
        if decl_str.startswith("static") and (context == "class" or context == "struct"):
            decl_str = decl_str[len("static"):].lstrip()
            static_method = True

        args_begin = decl_str.find("(")
        if decl_str.startswith("CVAPI"):
            rtype_end = decl_str.find(")", args_begin+1)
            if rtype_end < 0:
                print "Error at %d. no terminating ) in CVAPI() macro: %s" % (self.lineno, decl_str)
                sys.exit(-1)
            decl_str = decl_str[args_begin+1:rtype_end] + " " + decl_str[rtype_end+1:]
            args_begin = decl_str.find("(")
        if args_begin < 0:
            print "Error at %d: no args in '%s'" % (self.lineno, decl_str)
            sys.exit(-1)

        decl_start = decl_str[:args_begin].strip()
        # handle operator () case
        if decl_start.endswith("operator"):
            args_begin = decl_str.find("(", args_begin+1)
            if args_begin < 0:
                print "Error at %d: no args in '%s'" % (self.lineno, decl_str)
                sys.exit(-1)
            decl_start = decl_str[:args_begin].strip()
            # TODO: normalize all type of operators
            if decl_start.endswith("()"):
                decl_start = decl_start[0:-2].rstrip() + " ()"

        # constructor/destructor case
        if bool(re.match(r'(\w+::)*(?P<x>\w+)::~?(?P=x)', decl_start)):
            decl_start = "void " + decl_start

        rettype, funcname, modlist, argno = self.parse_arg(decl_start, -1)

        if argno >= 0:
            classname = top[1]
            if rettype == classname or rettype == "~" + classname:
                rettype, funcname = "", rettype
            else:
                if bool(re.match('\w+\s+\(\*\w+\)\s*\(.*\)', decl_str)):
                    return [] # function typedef
                elif bool(re.match('[A-Z_]+', decl_start)):
                    return [] # it seems to be a macro instantiation
                elif "__declspec" == decl_start:
                    return []
                else:
                    #print rettype, funcname, modlist, argno
                    print "Error at %s:%d the function/method name is missing: '%s'" % (self.hname, self.lineno, decl_start)
                    sys.exit(-1)

        if self.wrap_mode and (("::" in funcname) or funcname.startswith("~")):
            # if there is :: in function name (and this is in the header file),
            # it means, this is inline implementation of a class method.
            # Thus the function has been already declared within the class and we skip this repeated
            # declaration.
            # Also, skip the destructors, as they are always wrapped
            return []

        funcname = self.get_dotted_name(funcname)

        if not self.wrap_mode:
            decl = self.parse_func_decl_no_wrap(decl_str, static_method)
            decl[0] = funcname
            return decl

        arg_start = args_begin+1
        npos = arg_start-1
        balance = 1
        angle_balance = 0
        # scan the argument list; handle nested parentheses
        args_decls = []
        args = []
        argno = 1

        while balance > 0:
            npos += 1
            t, npos = self.find_next_token(decl_str, ["(", ")", ",", "<", ">"], npos)
            if not t:
                print "Error: no closing ')' at %d" % (self.lineno,)
                print decl_str
                print decl_str[arg_start:]
                sys.exit(-1)
            if t == "<":
                angle_balance += 1
            if t == ">":
                angle_balance -= 1
            if t == "(":
                balance += 1
            if t == ")":
                balance -= 1

            if (t == "," and balance == 1 and angle_balance == 0) or balance == 0:
                # process next function argument
                a = decl_str[arg_start:npos].strip()
                #print "arg = ", a
                arg_start = npos+1
                if a:
                    eqpos = a.find("=")
                    defval = ""
                    modlist = []
                    if eqpos >= 0:
                        defval = a[eqpos+1:].strip()
                    else:
                        eqpos = a.find("CV_DEFAULT")
                        if eqpos >= 0:
                            defval, pos3 = self.get_macro_arg(a, eqpos)
                        else:
                            eqpos = a.find("CV_WRAP_DEFAULT")
                            if eqpos >= 0:
                                defval, pos3 = self.get_macro_arg(a, eqpos)
                    if defval == "NULL":
                        defval = "0"
                    if eqpos >= 0:
                        a = a[:eqpos].strip()
                    arg_type, arg_name, modlist, argno = self.parse_arg(a, argno)
                    if self.wrap_mode:
                        if arg_type == "InputArray":
                            arg_type = "Mat"
                        elif arg_type == "InputOutputArray":
                            arg_type = "Mat"
                            modlist.append("/IO")
                        elif arg_type == "OutputArray":
                            arg_type = "Mat"
                            modlist.append("/O")
                        elif arg_type == "InputArrayOfArrays":
                            arg_type = "vector_Mat"
                        elif arg_type == "InputOutputArrayOfArrays":
                            arg_type = "vector_Mat"
                            modlist.append("/IO")
                        elif arg_type == "OutputArrayOfArrays":
                            arg_type = "vector_Mat"
                            modlist.append("/O")
                        defval = self.batch_replace(defval, [("InputArrayOfArrays", "vector<Mat>"),
                                                             ("InputOutputArrayOfArrays", "vector<Mat>"),
                                                             ("OutputArrayOfArrays", "vector<Mat>"),
                                                             ("InputArray", "Mat"),
                                                             ("InputOutputArray", "Mat"),
                                                             ("OutputArray", "Mat"),
                                                             ("noArray", arg_type)]).strip()
                    args.append([arg_type, arg_name, defval, modlist])
                npos = arg_start-1

        npos = decl_str.replace(" ", "").find("=0", npos)
        if npos >= 0:
            # skip pure virtual functions
            return []

        if static_method:
            func_modlist.append("/S")

        return [funcname, rettype, func_modlist, args]

    def get_dotted_name(self, name):
        """
        adds the dot-separated container class/namespace names to the bare function/class name, e.g. when we have

        namespace cv {
        class A {
        public:
            f(int);
        };
        }

        the function will convert "A" to "cv.A" and "f" to "cv.A.f".
        """
        if not self.block_stack:
            return name
        n = ""
        for b in self.block_stack:
            block_type, block_name = b[self.BLOCK_TYPE], b[self.BLOCK_NAME]
            if block_type in ["file", "enum"]:
                continue
            if block_type not in ["struct", "class", "namespace"]:
                print "Error at %d: there are non-valid entries in the current block stack " % (self.lineno, self.block_stack)
                sys.exit(-1)
            if block_name:
                n += block_name + "."
        return n + name.replace("::", ".")

    def parse_stmt(self, stmt, end_token):
        """
        parses the statement (ending with ';' or '}') or a block head (ending with '{')

        The function calls parse_class_decl or parse_func_decl when necessary. It returns
        <block_type>, <block_name>, <parse_flag>, <declaration>
        where the first 3 values only make sense for blocks (i.e. code blocks, namespaces, classes, enums and such)
        """
        stack_top = self.block_stack[-1]
        context = stack_top[self.BLOCK_TYPE]

        stmt_type = ""
        if end_token == "{":
            stmt_type = "block"

        if context == "block":
            print "Error at %d: should not call parse_stmt inside blocks" % (self.lineno,)
            sys.exit(-1)

        if context == "class" or context == "struct":
            while 1:
                colon_pos = stmt.find(":")
                if colon_pos < 0:
                    break
                w = stmt[:colon_pos].strip()
                if w in ["public", "protected", "private"]:
                    if w == "public":
                        stack_top[self.PUBLIC_SECTION] = True
                    else:
                        stack_top[self.PUBLIC_SECTION] = False
                    stmt = stmt[colon_pos+1:].strip()
                break

        # do not process hidden class members and template classes/functions
        if not stack_top[self.PUBLIC_SECTION] or stmt.startswith("template"):
            return stmt_type, "", False, None

        if end_token == "{":
            if not self.wrap_mode and stmt.startswith("typedef struct"):
                stmt_type = "struct"
                try:
                    classname, bases, modlist = self.parse_class_decl(stmt[len("typedef "):])
                except:
                    print "Error at %s:%d" % (self.hname, self.lineno)
                    exit(1)
                if classname.startswith("_Ipl"):
                    classname = classname[1:]
                decl = [stmt_type + " " + self.get_dotted_name(classname), "", modlist, []]
                if bases:
                    decl[1] = ": " + " ".join(bases)
                return stmt_type, classname, True, decl

            if stmt.startswith("class") or stmt.startswith("struct"):
                stmt_type = stmt.split()[0]
                if stmt.strip() != stmt_type:
                    try:
                        classname, bases, modlist = self.parse_class_decl(stmt)
                    except:
                        print "Error at %s:%d" % (self.hname, self.lineno)
                        exit(1)
                    decl = []
                    if ("CV_EXPORTS_W" in stmt) or ("CV_EXPORTS_AS" in stmt) or (not self.wrap_mode):# and ("CV_EXPORTS" in stmt)):
                        decl = [stmt_type + " " + self.get_dotted_name(classname), "", modlist, []]
                        if bases:
                            decl[1] = ": " + " ".join(bases)
                    return stmt_type, classname, True, decl

            if stmt.startswith("enum"):
                return "enum", "", True, None

            if stmt.startswith("namespace"):
                stmt_list = stmt.split()
                if len(stmt_list) < 2:
                    stmt_list.append("<unnamed>")
                return stmt_list[0], stmt_list[1], True, None
            if stmt.startswith("extern") and "\"C\"" in stmt:
                return "namespace", "", True, None

        if end_token == "}" and context == "enum":
            decl = self.parse_enum(stmt)
            return "enum", "", False, decl

        if end_token == ";" and stmt.startswith("typedef"):
            # TODO: handle typedef's more intelligently
            return stmt_type, "", False, None

        paren_pos = stmt.find("(")
        if paren_pos >= 0:
            # assume it's function or method declaration,
            # since we filtered off the other places where '(' can normally occur:
            #   - code blocks
            #   - function pointer typedef's
            decl = self.parse_func_decl(stmt)
            # we return parse_flag == False to prevent the parser to look inside function/method bodies
            # (except for tracking the nested blocks)
            return stmt_type, "", False, decl

        if (context == "struct" or context == "class") and end_token == ";" and stmt:
            # looks like it's member declaration; append the members to the class declaration
            class_decl = stack_top[self.CLASS_DECL]
            if ("CV_PROP" in stmt): # or (class_decl and ("/Map" in class_decl[2])):
                var_modlist = []
                if "CV_PROP_RW" in stmt:
                    var_modlist.append("/RW")
                stmt = self.batch_replace(stmt, [("CV_PROP_RW", ""), ("CV_PROP", "")]).strip()
                var_list = stmt.split(",")
                var_type, var_name1, modlist, argno = self.parse_arg(var_list[0], -1)
                var_list = [var_name1] + [i.strip() for i in var_list[1:]]

                for v in var_list:
                    class_decl[3].append([var_type, v, "", var_modlist])
            return stmt_type, "", False, None

        # something unknown
        return stmt_type, "", False, None

    def find_next_token(self, s, tlist, p=0):
        """
        Finds the next token from the 'tlist' in the input 's', starting from position 'p'.
        Returns the first occured token and its position, or ("", len(s)) when no token is found
        """
        token = ""
        tpos = len(s)
        for t in tlist:
            pos = s.find(t, p)
            if pos < 0:
                continue
            if pos < tpos:
                tpos = pos
                token = t
        return token, tpos

    def parse(self, hname, wmode=True):
        """
        The main method. Parses the input file.
        Returns the list of declarations (that can be print using print_decls)
        """
        self.hname = hname
        decls = []
        f = open(hname, "rt")
        linelist = list(f.readlines())
        f.close()

        # states:
        SCAN = 0 # outside of a comment or preprocessor directive
        COMMENT = 1 # inside a multi-line comment
        DIRECTIVE = 2 # inside a multi-line preprocessor directive

        state = SCAN

        self.block_stack = [["file", hname, True, True, None]]
        block_head = ""
        self.lineno = 0
        self.wrap_mode = wmode

        for l0 in linelist:
            self.lineno += 1
            #print self.lineno

            l = l0.strip()

            if state == SCAN and l.startswith("#"):
                state = DIRECTIVE
                # fall through to the if state == DIRECTIVE check

            if state == DIRECTIVE:
                if not l.endswith("\\"):
                    state = SCAN
                continue

            if state == COMMENT:
                pos = l.find("*/")
                if pos < 0:
                    continue
                l = l[pos+2:]
                state = SCAN

            if state != SCAN:
                print "Error at %d: invlid state = %d" % (self.lineno, state)
                sys.exit(-1)

            while 1:
                token, pos = self.find_next_token(l, [";", "\"", "{", "}", "//", "/*"])

                if not token:
                    block_head += " " + l
                    break

                if token == "//":
                    block_head += " " + l[:pos]
                    break

                if token == "/*":
                    block_head += " " + l[:pos]
                    pos = l.find("*/", pos+2)
                    if pos < 0:
                        state = COMMENT
                        break
                    l = l[pos+2:]
                    continue

                if token == "\"":
                    pos2 = pos + 1
                    while 1:
                        t2, pos2 = self.find_next_token(l, ["\\", "\""], pos2)
                        if t2 == "":
                            print "Error at %d: no terminating '\"'" % (self.lineno,)
                            sys.exit(-1)
                        if t2 == "\"":
                            break
                        pos2 += 2

                    block_head += " " + l[:pos2+1]
                    l = l[pos2+1:]
                    continue

                stmt = (block_head + " " + l[:pos]).strip()
                stmt = " ".join(stmt.split()) # normalize the statement
                stack_top = self.block_stack[-1]

                decl = None
                if stack_top[self.PROCESS_FLAG]:
                    # even if stack_top[PUBLIC_SECTION] is False, we still try to process the statement,
                    # since it can start with "public:"
                    stmt_type, name, parse_flag, decl = self.parse_stmt(stmt, token)
                    if decl:
                        if stmt_type == "enum":
                            for d in decl:
                                decls.append(d)
                        else:
                            decls.append(decl)
                else:
                    stmt_type, name, parse_flag = "block", "", False

                if token == "{":
                    if stmt_type == "class":
                        public_section = False
                    else:
                        public_section = True
                    self.block_stack.append([stmt_type, name, parse_flag, public_section, decl])

                if token == "}":
                    if not self.block_stack:
                        print "Error at %d: the block stack is empty" % (self.lineno,)
                    self.block_stack[-1:] = []
                    if pos+1 < len(l) and l[pos+1] == ';':
                        pos += 1

                block_head = ""
                l = l[pos+1:]

        return decls

    def print_decls(self, decls):
        """
        Prints the list of declarations, retrieived by the parse() method
        """
        for d in decls:
            print d[0], d[1], ";".join(d[2])
            for a in d[3]:
                print "   ", a[0], a[1], a[2],
                if a[3]:
                    print "; ".join(a[3])
                else:
                    print

if __name__ == '__main__':
    parser = CppHeaderParser()
    decls = []
    for hname in opencv_hdr_list:
        decls += parser.parse(hname)
    parser.print_decls(decls)
    print len(decls)
