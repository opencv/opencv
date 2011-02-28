import os, sys, re

finput=open(sys.argv[1], "rt")

# read the whole file content to s
s = "".join(finput.readlines())
finput.close()

# normalize line endings
s = re.sub(r"\r\n", "\n", s)

# remove trailing whitespaces
s = re.sub(r"[ \t]+\n", "\n", s)

# compress multiple empty lines
for i in range(5):
    s = re.sub(r"\n\n\n", "\n\n", s)

# remove empty line before ".." that terminates a code block
s = re.sub(r"\n\n\.\.\n", "\n..\n", s)

# move :: starting a code block to the end of previous line
s = re.sub(r"\n\n::\n", " ::\n", s)

# remove extra line breaks before/after _ or ,
s = re.sub(r"\n[ \t]*([_,])\n", r"\1", s)

# remove extra line breaks after `
s = re.sub(r"`\n", "` ", s)

# remove extra line breaks before `
s = re.sub(r"\n[ \t]*`", " `", s)

# remove links to wiki
s = re.sub(r"\n[ \t]*`id=\d[^`]+`__\n", "", s)

# remove trailing whitespaces one more time
s = re.sub(r"[ \t]+\n", "\n", s)

foutput=open(sys.argv[2], "wt")
foutput.write(s)
foutput.close()
