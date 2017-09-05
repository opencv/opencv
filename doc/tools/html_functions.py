from __future__ import print_function
import logging
import os
import codecs
from pprint import pprint

try:
    import bs4
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError('Error: '
                      'Install BeautifulSoup (bs4) for adding'
                      ' Python & Java signatures documentation')


def is_not_module_link(tmp_link):
    """ Checks if a link belongs to a c++ method """
    if tmp_link is None:
        return True
    if "group" not in tmp_link:
        return True
    if "#" in tmp_link:
        return True
    return False


def get_links_list(tmp_soup, filter_links):
    """ Get a list of links from a soup """
    tmp_href_list = []
    for tmp_link in tmp_soup.findAll('a'):
        tmp_href = tmp_link.get('href')
        if filter_links:
            if is_not_module_link(tmp_href):
                continue
        tmp_href_list.append(tmp_href)
    return tmp_href_list


def load_html_file(file_dir):
    """ Uses BeautifulSoup to load an html """
    with open(file_dir) as fp:
        tmp_soup = BeautifulSoup(fp, 'html.parser')
    return tmp_soup


def add_item(tmp_soup, new_row, is_parameter, text):
    """ Adds a new html tag for the table with the signature """
    new_item = tmp_soup.new_tag('td')
    if is_parameter:
        new_item = tmp_soup.new_tag('td', **{'class': 'paramname'})
    new_item.append(text)
    new_row.append(new_item)
    return new_row


def get_text_between_substrings(sig, begin_char, end_char):
    return sig.partition(begin_char)[-1].rpartition(end_char)[0]


def add_signature_to_table(tmp_soup, new_row, signature, function_name, language, ident):
    """ Add a signature to an html table"""
    if ident:
        new_item = tmp_soup.new_tag('td', style="padding-left: 0.5cm;")
    else:
        new_item = tmp_soup.new_tag('td')

    if str(signature.get('ret', None)) != "None":
        new_item.append(signature.get('ret') + ' =')
        new_row.append(new_item)

    if "Python" in language:
        pass  # function_name = "cv2." + function_name
    elif "Java" in language:
        # get word before function_name (= output)
        str_before_bracket = signature.split('(', 1)[0]
        list_of_words = str_before_bracket.split()
        output = list_of_words[len(list_of_words) - 2]
        new_item.append(output + " ")
        new_row.append(new_item)

    new_row = add_item(tmp_soup, new_row, False, signature.get('name', function_name) + '(')
    new_row = add_item(tmp_soup, new_row, True, signature['arg'])
    new_row = add_item(tmp_soup, new_row, False, ')')
    return new_row


def new_line(tmp_soup, tmp_table, new_row):
    """ Adds a new line to the html table """
    tmp_table.append(new_row)
    new_row = tmp_soup.new_tag('tr')
    return new_row


def add_bolded(tmp_soup, new_row, text):
    """ Adds bolded text to the table """
    new_item = tmp_soup.new_tag('th', style="text-align:left")
    new_item.append(text)
    new_row.append(new_item)
    return new_row


def create_description(tmp_soup, language, signatures, function_name):
    """ Insert the new Python / Java table after the current html c++ table """
    assert signatures
    tmp_table = tmp_soup.new_tag('table')
    new_row = tmp_soup.new_tag('tr')
    new_row = add_bolded(tmp_soup, new_row, language)
    ident = False

    new_row = new_line(tmp_soup, tmp_table, new_row)
    ident = True

    for s in signatures:
        new_row = new_line(tmp_soup, tmp_table, new_row)
        new_row = add_signature_to_table(tmp_soup, new_row, s, function_name, language, ident)
        new_row = new_line(tmp_soup, tmp_table, new_row)

    return tmp_table


def add_signatures(tmp_soup, tmp_dir, module_name, config):
    """ Add signatures to the current soup and rewrite the html file"""

    logging.debug(tmp_dir)
    sign_counter = 0
    python_sign_counter = 0
    java_sign_counter = 0

    if config.ADD_JAVA:
        functions_file = "java_doc_txts/" + module_name + "/functions.txt"
        if os.path.exists(functions_file):
            with open(functions_file, 'r') as f:
                java_signatures = f.read().split("\n")
        else:
            config.ADD_JAVA = False # This C++ module (module_name) may not exist in Java

    # the HTML tag & class being used to find functions
    for function in tmp_soup.findAll("h2", {"class": "memtitle"}):
        function_name = None
        for c in function.contents:
             if isinstance(c, bs4.element.NavigableString):
                 fn = str(c).encode("ascii","ignore").decode().strip()
                 if not fn.endswith('()'): # all functions have () in it's name
                     # enums, structures, etc
                     continue
                 function_name = fn[:-2]

        if not function_name:
            continue

        sign_counter += 1

        cpp_table = function.findNext('table')

        if config.ADD_PYTHON:
            signatures = config.python_signatures.get("cv::" + str(function_name), None)
            if signatures:
                print(function_name)

                description = create_description(tmp_soup, "Python:", signatures, function_name)
                description['class'] = 'python_language'
                old = cpp_table.next_sibling
                if old.name != 'table':
                    old = None
                elif not 'python_language' in old.get('class', []):
                    old = None
                if old is None:
                    cpp_table.insert_after(description)
                else:
                    old.replace_with(description)
                python_sign_counter += 1

        if config.ADD_JAVA:
            for signature in java_signatures:
                if function_name in signature:
                    create_description(cpp_table, tmp_soup, "Java:", signature, function_name)
                    java_sign_counter += 1
                    break

    tmp_str = str(tmp_soup)
    if os.name == 'nt': # if Windows
        with open(tmp_dir, "wb") as tmp_file:
            tmp_file.write(tmp_str.encode("ascii","ignore"))
    else:
        with open(tmp_dir, "w") as tmp_file:
            tmp_file.write(tmp_str)

    logging.debug("Added [" + str(python_sign_counter) + \
                  "/" + str(sign_counter) + "] Python signatures")
    logging.debug("Added [" + str(java_sign_counter) + \
                  "/" + str(sign_counter) + "] Java signatures")
