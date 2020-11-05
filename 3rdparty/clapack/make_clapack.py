appdoc = """
    This is generator of CLapack subset.
    The usage:

    1. Make sure you have the special version of f2c installed.
       Grab it from https://github.com/vpisarev/f2c/tree/for_lapack.
    2. Download fresh version of Lapack from
       https://github.com/Reference-LAPACK/lapack.
       You may choose some specific version or the latest snapshot.
    3. If necessary, edit "roots" and "banlist" variables in this script, specify the needed and unneeded functions
    4. From within a working directory run

       $ python3 <opencv_root>/3rdparty/clapack/make_clapack.py <lapack_root>
       or
       $ F2C=<path_to_custom_f2c> python3 <opencv_root>/3rdparty/clapack/make_clapack.py <lapack_root>

       it will generate "new_clapack" directory with "include" and "src" subdirectories.
    5. erase opencv/3rdparty/clapack/src and replace it with new_clapack/src.
    6. copy new_clapack/include/lapack.h to opencv/3rdparty/clapack/include.
    7. optionally, edit opencv/3rdparty/clapack/CMakeLists.txt and update CLAPACK_VERSION as needed.

    This is it. Now build it and enjoy.
"""

import glob, re, os, shutil, subprocess, sys

roots = ["cgemm_", "dgemm_", "sgemm_", "zgemm_",
         "dgeev_", "dgesdd_", 
         #"dsyevr_",
         #"dgesv_", "dgetrf_", "dposv_", "dpotrf_", "dgels_", "dgeqrf_",
         #"sgesv_", "sgetrf_", "sposv_", "spotrf_", "sgels_", "sgeqrf_"
         ]
banlist = ["slamch_", "slamc3_", "dlamch_", "dlamc3_", "lsame_", "xerbla_"]

if len(sys.argv) < 2:
    print(appdoc)
    sys.exit(0)

lapack_root = sys.argv[1]
dst_path = "."

def error(msg):
    print ("error: " + msg)
    sys.exit(0)

def file2fun(fname):
    return (os.path.basename(fname)[:-2]).upper()

def print_graph(m):
    for (k, neighbors) in sorted(m.items()):
        print (k + " : " + ", ".join(sorted(list(neighbors))))

blas_path = os.path.join(lapack_root, "BLAS/SRC")
lapack_path = os.path.join(lapack_root, "SRC")

roots = [f[:-1].upper() for f in roots]
banlist = [f[:-1].upper() for f in banlist]

def fun2file(func):
    filename = func.lower() + ".f"
    blas_loc = blas_path + "/" + filename
    lapack_loc = lapack_path + "/" + filename
    if os.path.exists(blas_loc):
        return blas_loc
    elif os.path.exists(lapack_loc):
        return lapack_loc
    else:
        error("neither %s nor %s exist" % (blas_loc, lapack_loc))

all_files = glob.glob(blas_path + "/*.f") + glob.glob(lapack_path + "/*.f")
all_funcs = [file2fun(fname) for fname in all_files]
all_funcs_set = set(all_funcs).difference(set(banlist))
all_funcs = sorted(list(all_funcs_set))

func_deps = {}

#print all_funcs

words_regexp = re.compile(r'\w+')

def scan_deps(func):
    global func_deps
    if func in func_deps:
        return
    func_deps[func] = set([]) # to avoid possibly infinite recursion
    f = open(fun2file(func), 'rt')
    deps = []
    external_mode = False
    for l in f.readlines():
        if l.startswith('*'):
            continue
        l = l.strip().upper()
        if l.startswith('EXTERNAL '):
            external_mode = True
        elif l.startswith('$') and external_mode:
            pass
        else:
            external_mode = False
        if not external_mode:
            continue
        for w in words_regexp.findall(l):
            if w in all_funcs_set:
                deps.append(w)
    f.close()
    # remove func from its dependencies
    deps = set(deps).difference(set([func]))
    func_deps[func] = deps
    for d in deps:
        scan_deps(d)

for r in roots:
    scan_deps(r)

selected_funcs = sorted(func_deps.keys())
print ("total files before amalgamation: %d" % len(selected_funcs))

inv_deps = {}
for func in selected_funcs:
    inv_deps[func] = set([])

for (func, deps) in func_deps.items():
    for d in deps:
        inv_deps[d] = inv_deps[d].union(set([func]))

#print_graph(inv_deps)

func_home = {}
for func in selected_funcs:
    func_home[func] = func

def get_home0(func, func0):
    used_by = inv_deps[func]
    if len(used_by) == 1:
        p = list(used_by)[0]
        if p != func and p != func0:
            return get_home0(p, func0)
        return func
    return func

# try to merge some files
for func in selected_funcs:
    func_home[func] = get_home0(func, func)

# try to merge some files even more
for iters in range(100):
    homes_changed = False
    for (func, used_by) in inv_deps.items():
        p0 = func_home[func]
        n = len(used_by)
        if n == 1:
            p = list(used_by)[0]
            p1 = func_home[p]
            if p1 != p0:
                func_home[func] = p1
                homes_changed = True
            continue
        elif n > 1:
            phomes = set([])
            for p in used_by:
                phomes.add(func_home[p])
            if len(phomes) == 1:
                p1 = list(phomes)[0]
                if p1 != p0:
                    func_home[func] = p1
                    homes_changed = True
    if not homes_changed:
        break

res_files = {}
for (func, h) in func_home.items():
    elems = res_files.get(h, set([]))
    elems.add(func)
    res_files[h] = elems

print ("total files after amalgamation: %d" % len(res_files))
#print_graph(res_files)

outdir = os.path.join(dst_path, "new_clapack")
outdir_src = os.path.join(outdir, "src")
outdir_inc = os.path.join(outdir, "include")

shutil.rmtree(outdir, ignore_errors=True)
try:
    os.makedirs(outdir_src)
except os.error:
    pass
try:
    os.makedirs(outdir_inc)
except os.error:
    pass

f2c_appname = os.getenv("F2C", default="f2c")
print ("f2c used: %s" % f2c_appname)

f2c_getver_cmd = f2c_appname + " -v"

verstr = subprocess.check_output(f2c_getver_cmd.split(' ')).decode("utf-8")
if "for_lapack" not in verstr:
    error("invalid version of f2c\n" + appdoc)

f2c_flags = "-ctypes -localconst -no-proto"
f2c_cmd0 = f2c_appname + " " + f2c_flags
f2c_cmd1 = f2c_appname + " -hdr none " + f2c_flags

lapack_protos = {}
extract_fn_regexp = re.compile(r'.+?(\w+)\s*\(')

def extract_proto(func, csrc):
    global lapack_protos
    cname = func.lower() + "_"
    cfname = func.lower() + ".c"
    regexp_str = r'\n(?:/\* Subroutine \*/\s*)?\w+\s+\w+\s*\((?:.|\n)+?\)[\s\n]*\{'
    proto_regexp = re.compile(regexp_str)
    ps = proto_regexp.findall(csrc)
    for p in ps:
        n = p.find("*/")
        if n < 0:
            n = 0
        else:
            n += 2
        p = p[n:-1].strip() + ";"
        fns = extract_fn_regexp.findall(p)
        if len(fns) != 1:
            error("prototype of function (%s) when analyzing %s cannot be parsed" % (p, cfname))
        fn = fns[0]
        if fn not in lapack_protos:
            p = re.sub(r'\bcomplex\b', 'lapack_complex', p)
            p = re.sub(r'\bdoublecomplex\b', 'lapack_doublecomplex', p)
            lapack_protos[fn] = p

for (filename, funcs) in sorted(res_files.items()):
    out = ""
    f2c_cmd = f2c_cmd0
    for func in sorted(list(funcs)):
        ffilename = fun2file(func)
        print ("running " + f2c_cmd + " on " + ffilename +  " ...")
        ffile = open(ffilename, 'rt')
        delta_out = subprocess.check_output(f2c_cmd.split(' '), stdin=ffile).decode("utf-8")
        # remove trailing whitespaces
        delta_out = '\n'.join([l.rstrip() for l in delta_out.split('\n')])
        extract_proto(func, delta_out)
        out += delta_out
        ffile.close()
        f2c_cmd = f2c_cmd1
    outname = os.path.join(outdir_src, filename.lower() + ".c")
    outfile = open(outname, 'wt')
    outfile.write(out)
    outfile.close()

proto_hdr = """// this is auto-generated header for Lapack subset
#ifndef __CLAPACK_H__
#define __CLAPACK_H__

#include "cblas.h"

#ifdef __cplusplus
extern "C" {
#endif

%s

#ifdef __cplusplus
}
#endif

#endif
""" % "\n\n".join([p for (n, p) in sorted(lapack_protos.items())])

proto_hdr_fname = os.path.join(outdir_inc, "lapack.h")
f = open(proto_hdr_fname, 'wt')
f.write(proto_hdr)
f.close()
