import logging
import os
import codecs
import cv2


try:
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

    if "-> None" in signature:
        pass
    elif "->" in signature:
        new_item.append(signature.split("->", 1)[1] + ' =')
        new_row.append(new_item)

    if "Python" in language:
        function_name = "cv2." + function_name
    elif "Java" in language:
        # get word before function_name (= output)
        str_before_bracket = signature.split('(', 1)[0]
        list_of_words = str_before_bracket.split()
        output = list_of_words[len(list_of_words) - 2]
        new_item.append(output + " ")
        new_row.append(new_item)

    new_row = add_item(tmp_soup, new_row, False, function_name + '(')
    new_row = add_item(tmp_soup, new_row, True, get_text_between_substrings(signature, "(", ")"))
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


def append_table_to(cpp_table, tmp_soup, language, signature, function_name):
    """ Insert the new Python / Java table after the current html c++ table """
    if signature != "":
        tmp_table = tmp_soup.new_tag('table')
        new_row = tmp_soup.new_tag('tr')
        new_row = add_bolded(tmp_soup, new_row, language)
        ident = False

        if len(signature) > 120:
            new_row = new_line(tmp_soup, tmp_table, new_row)
            ident = True

        if " or " in signature:
            ident = True
            for tmp_sig in signature.split(" or "):
                new_row = new_line(tmp_soup, tmp_table, new_row)
                new_row = add_signature_to_table(tmp_soup, new_row, tmp_sig, function_name, language, ident)
                new_row = new_line(tmp_soup, tmp_table, new_row)
        else:
            new_row = add_signature_to_table(tmp_soup, new_row, signature, function_name, language, ident)
            tmp_table.append(new_row)

        cpp_table.insert_after(tmp_table)
    return cpp_table


def add_signatures(tmp_soup, tmp_dir, ADD_JAVA, ADD_PYTHON, module_name):
    """ Add signatures to the current soup and rewrite the html file"""
    logging.debug(tmp_dir)
    sign_counter = 0
    python_sign_counter = 0
    java_sign_counter = 0

    if ADD_JAVA:
        functions_file = "java_doc_txts/" + module_name + "/functions.txt"
        if os.path.exists(functions_file):
            with open(functions_file, 'r') as f:
                java_signatures = f.read().split("\n")
        else:
            ADD_JAVA = False # This C++ module (module_name) may not exist in Java

    # the HTML tag & class being used to find functions
    for function in tmp_soup.findAll("h2", {"class": "memtitle"}):
        function_name = function.getText()
        if os.name == 'nt': # if Windows
            function_name = function_name.encode("ascii","ignore").decode()

        # all functions have () in it's name
        if "()" not in function_name:
            continue

        if "[" in function_name:
            if "[1/" in function_name:
                function_name = function_name.replace(' ', '')[:-7]
            else:
                continue
        else:
            function_name = function_name.replace(' ', '')[:-2]
        sign_counter += 1

        # if not Windows computer
        if os.name != 'nt':
            function_name = function_name.replace(' ', '')[2:]

        cpp_table = function.findNext('table')

        if ADD_PYTHON:
            try:
                print(function_name)
                method = getattr(cv2, str(function_name))
                description = str(method.__doc__).split("\n")
                signature = ""
                is_first_sig = True
                for line in description:
                    if line.startswith(".") or line == "":
                        continue
                    else:
                        if is_first_sig:
                            signature += line
                            is_first_sig = False
                        else:
                            signature += " or " + line

                cpp_table = append_table_to(cpp_table, tmp_soup, "Python:", signature, function_name)
                python_sign_counter += 1
            except AttributeError:
                continue

        if ADD_JAVA:
            for signature in java_signatures:
                if function_name in signature:
                    append_table_to(cpp_table, tmp_soup, "Java:", signature, function_name)
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
