import sys

f=open(sys.argv[1], "rt")
ll = list(f.readlines())
f.close()
f=open(sys.argv[1], "wt")
singleparam = False

for l in ll:
    l = l.replace("\\code{~const}}{}", "}{\\code{~const}}")
    if l.startswith("\\item[{Parameters}] \\leavevmode"):
        if not l.startswith("\\item[{Parameters}] \\leavevmode\\begin{itemize}"):
            singleparam = True
        l = "\\item[{Parameters}] \\leavevmode\\begin{itemize}[label=]\n"
        if singleparam:
            l += "\\item {}\n"
    elif singleparam and l.startswith("\\end{description}\\end{quote}"):
        l = "\\end{itemize}\n" + l
        singleparam = False
    f.write(l)

f.close()
