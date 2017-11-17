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

def add_signature_to_table(soup, tmp_row, signature, language):
    """ Add a signature to an html table"""
    new_item = soup.new_tag('td', style="padding-left: 0.5cm;")

    if str(signature.get('ret', None)) != "None":
        new_item.append(signature.get('ret') + ' =')
        tmp_row.append(new_item)

    tmp_row, soup = add_item(soup, tmp_row, False, "cv2." + signature.get('name', None) + '(')
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


def create_description(soup, language, signatures):
    """ Insert the new Python / Java table after the current html c++ table """
    assert signatures
    tmp_table = soup.new_tag('table')
    new_row = soup.new_tag('tr')
    new_row, soup = add_bolded(soup, new_row, language)
    new_row, soup = new_line(soup, tmp_table, new_row)
    for s in signatures:
        new_row, soup = new_line(soup, tmp_table, new_row)
        new_row, soup = add_signature_to_table(soup, new_row, s, language)
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

def append_python_signatures_to_table(soup, signatures, table):
    description, soup = create_description(soup, "Python:", signatures)
    description['class'] = 'python_language'
    old = table.next_sibling
    if old.name != 'table':
        old = None
    elif not 'python_language' in old.get('class', []):
        old = None
    # if already existed replace with the new
    if old is None:
        table.insert_after(description)
    else:
        old.replace_with(description)
    table.insert_after(description)
    return soup

def get_heading_text(a):
    str = ""
    element = a.parent.parent
    if element is not None:
        if element.has_attr('class'):
            tmp_class = element["class"][0]
            if "memitem:" in tmp_class and "python" not in tmp_class:
                str = element.parent.find("tr").text
    return str

def append_python_signatures_to_heading(soup, signatures, element, href):
    for s in signatures:
        attrs = {'class': 'memitem:python'}
        new_tr = soup.new_tag('tr', **attrs)

        attrs_left = {'class': 'memItemLeft', 'valign': 'top', 'align': 'right'}
        new_td_left = soup.new_tag('td', **attrs_left)
        new_td_left.append(str(s.get('ret', None)))
        new_tr.append(new_td_left)

        attrs_right = {'class': 'memItemRight', 'valign': 'bottom'}
        new_td_right = soup.new_tag('td', **attrs_right)
        attrs_a = {'class': 'el', 'href': href}
        new_a = soup.new_tag('a', **attrs_a)
        new_a.append('cv2.' + str(s.get('name', None)))
        new_td_right.append(new_a)
        new_td_right.append("(" + s['arg'] +")")

        new_tr.append(new_td_right)

        old = element.next_sibling
        if old is not None:
            if old.name != 'tr':
                old = None
            elif not 'memitem:python' in old.get('class', []):
                old = None
        # if already existed replace with the new
        if old is None:
            element.insert_after(new_tr)
        else:
            old.replace_with(new_tr)
    return soup

def append_python_signature(function_variants, anchor_list, soup):
    type = anchor_list[0].type
    if type == "fn":
        if len(anchor_list) == 1:
            tmp_anchor = anchor_list[0].anchor
            a_list = get_anchor_list(tmp_anchor, soup)
            for a in a_list:
                if a['href'] == "#" + tmp_anchor:
                    # Function Documentation (tables)
                    table = a.findNext('table')
                    if table is not None:
                        soup = append_python_signatures_to_table(soup, function_variants, table)
                    continue
                str = get_heading_text(a)
                if "Functions" in str:
                    soup = append_python_signatures_to_heading(soup, function_variants, a.parent, a['href'])
                    print("One more")
    return soup

def update_html(file, soup):
    tmp_str = str(soup)
    if os.name == 'nt': # if Windows
        with open(file, "wb") as tmp_file:
            tmp_file.write(tmp_str.encode("ascii","ignore"))
    else:
        with open(file, "w") as tmp_file:
            tmp_file.write(tmp_str)
