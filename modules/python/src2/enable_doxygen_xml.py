import re
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    text = f.read()

text = re.sub(r'(\s*GENERATE_XML\s*=\s*)NO', r'\1YES', text)

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(text)
