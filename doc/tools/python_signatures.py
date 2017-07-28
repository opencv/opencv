"""
This code adds Python signatures to the docs.

TODO:
* clarify when there are several C++ signatures corresponding to a single Python function.
    i.e: calcHist():
    http://docs.opencv.org/3.2.0/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d
* clarify special case:
    http://docs.opencv.org/3.2.0/db/de0/group__core__utils.html#ga4910d7f86336cd4eff9dd05575667e41
"""
import re
import sys
import html_functions

ADD_JAVA = False
ADD_PYTHON = True
ROOT_DIR = sys.argv[1]

soup = html_functions.load_html_file(ROOT_DIR + "index.html")
href_list = html_functions.get_links_list(soup, True)

for link in href_list:
    # add python signatures to the module
    soup = html_functions.load_html_file(ROOT_DIR + link)
    sub_href_list = html_functions.get_links_list(soup, True)
    module_name = html_functions.get_text_between_substrings(link, "group__", ".html")
    html_functions.add_signatures(soup, ROOT_DIR + link, ADD_JAVA, ADD_PYTHON, module_name)

    # add python signatures to the sub-modules
    link = re.sub(r"group__.+html", "", link)
    for sub_link in sub_href_list:
        tmp_dir = ROOT_DIR + link + sub_link
        soup = html_functions.load_html_file(tmp_dir)
        html_functions.add_signatures(soup, tmp_dir, ADD_JAVA, ADD_PYTHON, module_name)
