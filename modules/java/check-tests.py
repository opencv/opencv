import sys, os, re

class JavaParser:
    def __init__(self):
        self.clear()

    def clear(self):
        self.mdict = {}
        self.tdict = {}
        self.r1 = re.compile("\s*public\s+(?:static\s+)?(\w+)\(([^)]*)\)") # c-tor
        self.r2 = re.compile("\s*public\s+(?:static\s+)?\w+\s+(\w+)\(([^)]*)\)")

    def dict2set(self, d):
        s = set()
        for f in d.keys():
            if len(d[f]) == 1:
                s.add(f)
            else:
                s |= set(d[f])
        return s


    def get_not_tested(self):
        mset = self.dict2set(self.mdict)
        tset = self.dict2set(self.tdict)
        return mset - tset


    def parse(self, path):
        if ".svn" in path:
            return
        if os.path.isfile(path):
            parser.parse_file(path)
        elif os.path.isdir(path):
            for x in os.listdir(path):
                self.parse(path + "/" + x)
        return


    def parse_file(self, fname):
        clsname = os.path.basename(fname).replace("Test", "").replace(".java", "")
        clsname = clsname[0].upper() + clsname[1:]
        f = open(fname, "rt")
        for line in f:
            m1 = self.r1.match(line)
            m2 = self.r2.match(line)
            func = ''
            args_str = ''
            if m1:
                func = m1.group(1)
                args_str = m1.group(2)
            elif m2:
                func = m2.group(1)
                args_str = m2.group(2)
            else:
                continue
            d = (self.mdict, self.tdict)["test" in func]
            func = re.sub(r"^test", "", func)
            func = clsname + "--" + func[0].upper() + func[1:]
            args_str = args_str.replace("[]", "Array").replace("...", "Array ")
            args_str = re.sub(r"List<(\w+)>", "ListOf\g<1>", args_str)
            args = [a.split()[0] for a in args_str.split(",") if a]
            func_ex = func + "".join([a[0].upper() + a[1:] for a in args])
            if func in d:
                d[func].append(func_ex)
            else:
                d[func] = [func_ex]
        f.close()
        return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage:\n", \
            os.path.basename(sys.argv[0]), \
            "<Classes/Tests dir1/file1> [<Classes/Tests dir2/file2> ...]\n", "Not tested methods are logged to stdout."
        exit(0)
    parser = JavaParser()
    for x in sys.argv[1:]:
        parser.parse(x)
    funcs = parser.get_not_tested()
    if funcs:
        print "UNTESTED methods (%i):\n\t" % len(funcs), "\n\t".join(sorted(funcs))
    print "Done."

