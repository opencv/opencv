from __future__ import print_function
import logging
import os
from pprint import pprint

try:
    import bs4
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError('Error: '
                      'Install BeautifulSoup (bs4) for adding'
                      ' Python & Java signatures documentation')

def load_html_file(file_dir):
    """ Uses BeautifulSoup to load an html """
    with open(file_dir) as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    return soup

def add_item(soup, new_row, is_parameter, text):
    """ Adds a new html tag for the table with the signature """
    new_item = soup.new_tag('td')
    if is_parameter:
        new_item = soup.new_tag('td', **{'class': 'paramname'})
    new_item.append(text)
    new_row.append(new_item)
    return new_row, soup

def add_signature_to_table(soup, tmp_row, signature, language, type):
    """ Add a signature to an html table"""
    new_item = soup.new_tag('td', style="padding-left: 0.5cm;")

    if str(signature.get('ret', None)) != "None":
        new_item.append(signature.get('ret') + ' =')
    tmp_row.append(new_item)

    tmp_name = signature.get('name', None)
    if type is not "method":
        tmp_name = "cv2." + tmp_name
    else:
        tmp_name = "obj." + tmp_name
    tmp_row, soup = add_item(soup, tmp_row, False, tmp_name + '(')
    tmp_row, soup = add_item(soup, tmp_row, True, signature['arg'])
    tmp_row, soup = add_item(soup, tmp_row, False, ')')
    return tmp_row, soup


def new_line(soup, tmp_table, new_row):
    """ Adds a new line to the html table """
    tmp_table.append(new_row)
    new_row = soup.new_tag('tr')
    return new_row, soup


def add_bolded(soup, new_row, text):
    """ Adds bolded text to the table """
    new_item = soup.new_tag('th', style="text-align:left")
    new_item.append(text)
    new_row.append(new_item)
    return new_row, soup


def create_description(soup, language, signatures, type):
    """ Insert the new Python / Java table after the current html c++ table """
    assert signatures
    tmp_table = soup.new_tag('table')
    new_row = soup.new_tag('tr')
    new_row, soup = add_bolded(soup, new_row, language)
    new_row, soup = new_line(soup, tmp_table, new_row)
    for s in signatures:
        new_row, soup = new_line(soup, tmp_table, new_row)
        new_row, soup = add_signature_to_table(soup, new_row, s, language, type)
        new_row, soup = new_line(soup, tmp_table, new_row)
    return tmp_table, soup


def get_anchor_list(anchor, soup):
    a_list = []
    # go through all the links
    for a in soup.find_all('a', href=True):
        # find links with the same anchor
        last_part_of_link = a['href'].rsplit('#', 1)[-1]
        if last_part_of_link == anchor:
            a_list.append(a)
    return a_list

def is_static_method(element):
    if element.name == "table":
        tmp_element = element.find('td', {'class': 'memname'})
        if tmp_element is not None:
            if 'static' in tmp_element.text:
                return True
    else:
        if element['class'][0] == 'memItemRight':
           if "static" in element.previousSibling.text:
               return True
    return False

def append_python_signatures_to_table(soup, signatures, table, type):
    if type == "method":
        if is_static_method(table):
            type = "static" + type
    description, soup = create_description(soup, "Python:", signatures, type)
    description['class'] = 'python_language'
    soup = insert_or_replace(soup, table, description, "table", "python_language")
    return soup

def get_heading_text(a):
    str = ""
    element = a.parent
    if element is not None:
        childs = element.find_all('a')
        # the anchor should not be an argument of a function / method
        if childs.index(a) is not 0:
            return str
    element = element.parent
    if element is not None:
        if element.has_attr('class'):
            tmp_class = element["class"][0]
            if "memitem:" in tmp_class and "python" not in tmp_class:
                str = element.parent.find("tr").text
    return str

def insert_or_replace(soup, element, description, tag_name, tag_class):
    old = element.next_sibling
    if old is not None:
        if old.name != tag_name:
            old = None
        elif not tag_class in old.get('class', []):
            old = None
    # if already existed replace with the new
    if old is None:
        element.insert_after(description)
    else:
        old.replace_with(description)
    return soup

def new_heading_td(soup, s, href, type):
    if href is None:
        attrs = {'class': 'memItemLeft', 'valign': 'top', 'align': 'right'}
        new_td = soup.new_tag('td', **attrs)
        new_td.append(str(s.get('ret', None)))
    else:
        attrs = {'class': 'memItemRight', 'valign': 'bottom'}
        new_td = soup.new_tag('td', **attrs)

        # make the function name linkable
        attrs_a = {'class': 'el', 'href': href}
        new_a = soup.new_tag('a', **attrs_a)
        tmp_name = str(s.get('name', None))
        if type is not "method":
            tmp_name = "cv2." + tmp_name
        else:
            tmp_name = "obj." + tmp_name
        new_a.append(tmp_name)
        new_td.append(new_a)

        new_td.append("(" + s['arg'] +")")
    return soup, new_td

def append_python_signatures_to_heading(soup, signatures, element, href, type):
    if type == "method":
        if is_static_method(element):
            type = "static" + type
    for s in signatures:
        attrs = {'class': 'memitem:python'}
        new_tr = soup.new_tag('tr', **attrs)

        soup, new_td_left = new_heading_td(soup, s, None, type)
        new_tr.append(new_td_left)

        soup, new_td_right = new_heading_td(soup, s, href, type)
        new_tr.append(new_td_right)

        soup = insert_or_replace(soup, element, new_tr, "tr", "memitem:python")
    return soup

def append_python_signature(function_variants, anchor_list, soup):
    type = anchor_list[0].type
    if type == "method" or type == "fn":
        if len(anchor_list) == 1:
            tmp_anchor = anchor_list[0].anchor
            a_list = get_anchor_list(tmp_anchor, soup)
            for a in a_list:
                if a['href'] == "#" + tmp_anchor:
                    tmp_element = a.parent
                    # ignore the More... link <td class = mdescRight>
                    if tmp_element is None or tmp_element['class'][0] == 'mdescRight':
                        continue
                    # Function Documentation (tables)
                    table = a.findNext('table')
                    if table is not None:
                        soup = append_python_signatures_to_table(soup, function_variants, table, type)
                else:
                    str = get_heading_text(a)
                    if "Functions" in str:
                        soup = append_python_signatures_to_heading(soup, function_variants, a.parent, a['href'], type)
    return soup

def update_html(file, soup):
    tmp_str = str(soup)
    if os.name == 'nt': # if Windows
        with open(file, "wb") as tmp_file:
            tmp_file.write(tmp_str.encode("ascii","ignore"))
    else:
        with open(file, "w") as tmp_file:
            tmp_file.write(tmp_str)
