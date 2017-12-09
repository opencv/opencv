from __future__ import print_function
import sys

import logging
import os
from pprint import pprint
import traceback

try:
    import bs4
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError('Error: '
                      'Install BeautifulSoup (bs4) for adding'
                      ' Python & Java signatures documentation')

def load_html_file(file_dir):
    """ Uses BeautifulSoup to load an html """
    with open(file_dir, 'rb') as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    return soup

def update_html(file, soup):
    s = str(soup)
    if os.name == 'nt' or sys.version_info[0] == 3: # if Windows
        s = s.encode('utf-8', 'ignore')
    with open(file, 'wb') as f:
        f.write(s)


def insert_python_signatures(python_signatures, symbols_dict, filepath):
    soup = load_html_file(filepath)
    entries = soup.find_all(lambda tag: tag.name == "a" and tag.has_attr('id'))
    for e in entries:
        anchor = e['id']
        if anchor in symbols_dict:
            s = symbols_dict[anchor]
            logging.info('Process: %r' % s)
            if s.type == 'fn' or s.type == 'method':
                process_fn(soup, e, python_signatures[s.cppname], s)
            elif s.type == 'const':
                process_const(soup, e, python_signatures[s.cppname], s)
            else:
                logging.error('unsupported type: %s' % s);

    update_html(filepath, soup)


def process_fn(soup, anchor, python_signature, symbol):
    try:
        r = anchor.find_next_sibling(class_='memitem').find(class_='memproto').find('table')
        insert_python_fn_signature(soup, r, python_signature, symbol)
    except:
        logging.error("Can't process: %s" % symbol)
        traceback.print_exc()
        pprint(anchor)


def process_const(soup, anchor, python_signature, symbol):
    try:
        #pprint(anchor.parent)
        description = append(soup.new_tag('div', **{'class' : ['python_language']}),
            'Python: ' + python_signature[0]['name'])
        old = anchor.find_next_sibling('div', class_='python_language')
        if old is None:
            anchor.parent.append(description)
        else:
            old.replace_with(description)
        #pprint(anchor.parent)
    except:
        logging.error("Can't process: %s" % symbol)
        traceback.print_exc()
        pprint(anchor)


def insert_python_fn_signature(soup, table, variants, symbol):
    description = create_python_fn_description(soup, variants)
    description['class'] = 'python_language'
    soup = insert_or_replace(table, description, 'table', 'python_language')
    return soup


def create_python_fn_description(soup, variants):
    language = 'Python:'
    table = soup.new_tag('table')
    heading_row = soup.new_tag('th')
    table.append(
        append(soup.new_tag('tr'),
               append(soup.new_tag('th', colspan=999, style="text-align:left"), language)))
    for v in variants:
        #logging.debug(v)
        add_signature_to_table(soup, table, v, language, type)
    #print(table)
    return table


def add_signature_to_table(soup, table, signature, language, type):
    """ Add a signature to an html table"""
    row = soup.new_tag('tr')
    row.append(soup.new_tag('td', style='width: 20px;'))

    if 'ret' in signature:
        row.append(append(soup.new_tag('td'), signature['ret']))
        row.append(append(soup.new_tag('td'), '='))
    else:
        row.append(soup.new_tag('td')) # return values
        row.append(soup.new_tag('td')) # '='

    row.append(append(soup.new_tag('td'), signature['name'] + '('))
    row.append(append(soup.new_tag('td', **{'class': 'paramname'}), signature['arg']))
    row.append(append(soup.new_tag('td'), ')'))
    table.append(row)


def append(target, obj):
    target.append(obj)
    return target


def insert_or_replace(element_before, new_element, tag, tag_class):
    old = element_before.find_next_sibling(tag, class_=tag_class)
    if old is None:
        element_before.insert_after(new_element)
    else:
        old.replace_with(new_element)
