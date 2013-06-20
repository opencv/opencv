#!/usr/bin/env python

from __future__ import division

import ast
import logging
import numbers
import os, os.path
import re

from argparse import ArgumentParser
from collections import OrderedDict
from glob import glob
from itertools import ifilter

import xlwt

from testlog_parser import parseLogFile

# To build XLS report you neet to put your xmls (OpenCV tests output) in the
# following way:
#
# "root" --- folder, representing the whole XLS document. It contains several
# subfolders --- sheet-paths of the XLS document. Each sheet-path contains it's
# subfolders --- config-paths. Config-paths are columns of the sheet and
# they contains xmls files --- output of OpenCV modules testing.
# Config-path means OpenCV build configuration, including different
# options such as NEON, TBB, GPU enabling/disabling.
#
# root
# root\sheet_path
# root\sheet_path\configuration1 (column 1)
# root\sheet_path\configuration2 (column 2)

re_image_size = re.compile(r'^ \d+ x \d+$', re.VERBOSE)
re_data_type = re.compile(r'^ (?: 8 | 16 | 32 | 64 ) [USF] C [1234] $', re.VERBOSE)

time_style = xlwt.easyxf(num_format_str='#0.00')
no_time_style = xlwt.easyxf('pattern: pattern solid, fore_color gray25')

speedup_style = time_style
good_speedup_style = xlwt.easyxf('font: color green', num_format_str='#0.00')
bad_speedup_style = xlwt.easyxf('font: color red', num_format_str='#0.00')
no_speedup_style = no_time_style
error_speedup_style = xlwt.easyxf('pattern: pattern solid, fore_color orange')
header_style = xlwt.easyxf('font: bold true; alignment: horizontal centre, vertical top, wrap True')

def collect_xml(collection, configuration, xml_fullname):
    xml_fname = os.path.split(xml_fullname)[1]
    module = xml_fname[:xml_fname.index('_')]

    module_tests = collection.setdefault(module, OrderedDict())

    for test in sorted(parseLogFile(xml_fullname)):
        test_results = module_tests.setdefault((test.shortName(), test.param()), {})
        test_results[configuration] = test.get("gmean") if test.status == 'run' else test.status

def main():
    arg_parser = ArgumentParser(description='Build an XLS performance report.')
    arg_parser.add_argument('sheet_dirs', nargs='+', metavar='DIR', help='directory containing perf test logs')
    arg_parser.add_argument('-o', '--output', metavar='XLS', default='report.xls', help='name of output file')
    arg_parser.add_argument('-c', '--config', metavar='CONF', help='global configuration file')

    args = arg_parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    if args.config is not None:
        with open(args.config) as global_conf_file:
            global_conf = ast.literal_eval(global_conf_file.read())
    else:
        global_conf = {}

    wb = xlwt.Workbook()

    for sheet_path in args.sheet_dirs:
        try:
            with open(os.path.join(sheet_path, 'sheet.conf')) as sheet_conf_file:
                sheet_conf = ast.literal_eval(sheet_conf_file.read())
        except Exception:
            sheet_conf = {}
            logging.debug('no sheet.conf for %s', sheet_path)

        sheet_conf = dict(global_conf.items() + sheet_conf.items())

        if 'configurations' in sheet_conf:
            config_names = sheet_conf['configurations']
        else:
            try:
                config_names = [p for p in os.listdir(sheet_path)
                    if os.path.isdir(os.path.join(sheet_path, p))]
            except Exception as e:
                logging.warning('error while determining configuration names for %s: %s', sheet_path, e)
                continue

        collection = {}

        for configuration, configuration_path in \
                [(c, os.path.join(sheet_path, c))  for c in config_names]:
            logging.info('processing %s', configuration_path)
            for xml_fullname in glob(os.path.join(configuration_path, '*.xml')):
                collect_xml(collection, configuration, xml_fullname)

        sheet = wb.add_sheet(sheet_conf.get('sheet_name', os.path.basename(os.path.abspath(sheet_path))))

        sheet.row(0).height = 800
        sheet.panes_frozen = True
        sheet.remove_splits = True
        sheet.horz_split_pos = 1
        sheet.horz_split_first_visible = 1

        sheet_comparisons = sheet_conf.get('comparisons', [])

        for i, w in enumerate([2000, 15000, 2500, 2000, 15000]
                + (len(config_names) + 1 + len(sheet_comparisons)) * [3000]):
            sheet.col(i).width = w

        for i, caption in enumerate(['Module', 'Test', 'Image\nsize', 'Data\ntype', 'Parameters']
                + config_names + [None]
                + [comp['to'] + '\nvs\n' + comp['from'] for comp in sheet_comparisons]):
            sheet.row(0).write(i, caption, header_style)

        row = 1

        module_colors = sheet_conf.get('module_colors', {})
        module_styles = {module: xlwt.easyxf('pattern: pattern solid, fore_color {}'.format(color))
                         for module, color in module_colors.iteritems()}

        for module, tests in sorted(collection.iteritems()):
            for ((test, param), configs) in tests.iteritems():
                sheet.write(row, 0, module, module_styles.get(module, xlwt.Style.default_style))
                sheet.write(row, 1, test)

                param_list = param[1:-1].split(", ")
                sheet.write(row, 2, next(ifilter(re_image_size.match, param_list), None))
                sheet.write(row, 3, next(ifilter(re_data_type.match, param_list), None))

                sheet.row(row).write(4, param)
                for i, c in enumerate(config_names):
                    if c in configs:
                        sheet.write(row, 5 + i, configs[c], time_style)
                    else:
                        sheet.write(row, 5 + i, None, no_time_style)

                for i, comp in enumerate(sheet_comparisons):
                    cmp_from = configs.get(comp["from"])
                    cmp_to = configs.get(comp["to"])
                    col = 5 + len(config_names) + 1 + i

                    if isinstance(cmp_from, numbers.Number) and isinstance(cmp_to, numbers.Number):
                        try:
                            speedup = cmp_from / cmp_to
                            sheet.write(row, col, speedup, good_speedup_style if speedup > 1.1 else
                                                           bad_speedup_style  if speedup < 0.9 else
                                                           speedup_style)
                        except ArithmeticError as e:
                            sheet.write(row, col, None, error_speedup_style)
                    else:
                        sheet.write(row, col, None, no_speedup_style)

                row += 1
                if row % 1000 == 0: sheet.flush_row_data()

    wb.save(args.output)

if __name__ == '__main__':
    main()
