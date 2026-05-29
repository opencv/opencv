"""Profile what's eating space in a Sphinx environment.pickle.

Run: python inspect_env_pickle.py <path-to-environment.pickle>
"""
import pickle, sys, os, pathlib, types
from collections import defaultdict

PICKLE = sys.argv[1] if len(sys.argv) > 1 else \
    "/media/abhishek/hugedrive21/opencv_abhishek/build/docs_sphinx/html/.doctrees/environment.pickle"


def _pickled_size(obj) -> int:
    """How many bytes does this object take when pickled? Cheap proxy for
    'how much does this contribute to the env.pickle on disk'."""
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as e:
        return -1


def _fmt(n: int) -> str:
    if n < 0: return "n/a"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


print(f"Loading {PICKLE} ({_fmt(os.path.getsize(PICKLE))}) — this takes a while…")
with open(PICKLE, "rb") as f:
    env = pickle.load(f)
print("Loaded.\n")

# Top-level attributes by pickled size.
attrs = [(a, getattr(env, a)) for a in dir(env)
         if not a.startswith("_") and not callable(getattr(env, a, None))]
sized = [(name, val, _pickled_size(val)) for name, val in attrs]
sized.sort(key=lambda x: x[2], reverse=True)

total = sum(s for _, _, s in sized if s > 0)
print(f"{'attr':<30} {'size':>10}   type / sample")
print("-" * 80)
for name, val, sz in sized[:30]:
    typename = type(val).__name__
    if isinstance(val, dict):
        sample = f"dict[{len(val)} keys]"
    elif isinstance(val, (set, list, tuple)):
        sample = f"{typename}[{len(val)} items]"
    else:
        sample = typename
    print(f"{name:<30} {_fmt(sz):>10}   {sample}")
print(f"\nsum of top-level attrs: {_fmt(total)}\n")

# Drill into domains — usually the elephant.
domains = getattr(env, "domains", None)
if domains:
    print("Per-domain pickle size:")
    dom_sized = [(n, _pickled_size(d)) for n, d in domains.items()]
    dom_sized.sort(key=lambda x: x[1], reverse=True)
    for n, sz in dom_sized:
        print(f"  {n:<20} {_fmt(sz):>10}")
    print()

# Drill into C++ domain specifically — known huge for breathe-heavy projects.
cpp = getattr(env, "domains", {}).get("cpp")
if cpp is not None:
    print("C++ domain attribute sizes:")
    cpp_attrs = [(a, getattr(cpp, a)) for a in dir(cpp)
                 if not a.startswith("_") and not callable(getattr(cpp, a, None))]
    cpp_sized = [(name, val, _pickled_size(val)) for name, val in cpp_attrs]
    cpp_sized.sort(key=lambda x: x[2], reverse=True)
    for name, val, sz in cpp_sized[:15]:
        if sz < 1024 * 1024:
            continue
        sample = f"{type(val).__name__}"
        if isinstance(val, dict):
            sample += f"[{len(val)} keys]"
        elif isinstance(val, (list, set, tuple)):
            sample += f"[{len(val)} items]"
        print(f"  {name:<30} {_fmt(sz):>10}   {sample}")
    # The CPPDomain root_symbol is typically the giant one — count its descendants.
    root = getattr(cpp, "root_symbol", None)
    if root is not None:
        # CPPDomain.Symbol is a tree; count nodes recursively.
        try:
            def count(s):
                n = 1
                for c in getattr(s, "_children", []) or []:
                    n += count(c)
                return n
            n_syms = count(root)
            print(f"\n  CPP symbol tree node count: {n_syms}")
        except Exception as e:
            print(f"  (symbol tree walk failed: {e})")
    print()

# Doctrees in env.all_doctrees? Sphinx stores doctrees on disk in .doctrees/*.doctree,
# NOT in env.pickle. But env.titles, env.tocs etc. duplicate parts.
print("Sphinx-wise: doctrees themselves live in <outdir>/.doctrees/*.doctree, not pickle.")
print("If env.pickle dominates load time, the C++ domain symbol tree is usually the cause.")
