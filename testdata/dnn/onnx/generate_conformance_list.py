import sys
import os

from contextlib import redirect_stdout  # Python 3.4+

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'model.onnx'
DATA_DIR = 'test_data_set_0'

TEST_DATA_PATH = None #os.environ.get('OPENCV_TEST_DATA_PATH', None)
if TEST_DATA_PATH is not None:
    ROOT_DIR = os.path.join(TEST_DATA_PATH, 'dnn/onnx/conformance')
else:
    ROOT_DIR = os.path.join(SCRIPT_PATH, 'conformance')

NODE_DIR = os.path.join(ROOT_DIR, 'node')

def dump_test_list():
    print('--- This content is a stub for test_onnx_conformance.cpp ---')
    print('static const TestCase testConformanceConfig[] =')
    print('{')
    for test_name in os.listdir(NODE_DIR):
        test_path = os.path.join(NODE_DIR, test_name)

        assert os.path.isdir(test_path), 'node folder should contain only directories'
        children = sorted([x for x in os.listdir(test_path)])
        assert children == [MODEL_NAME, DATA_DIR], 'test folder should contain model and one dataset'

        data_prefix = os.path.join(DATA_DIR)
        dataset_path = os.path.join(test_path, DATA_DIR)

        inputs = 0
        outputs = 0
        for data_name in os.listdir(dataset_path):
            data_path = os.path.join(data_prefix, data_name)
            if data_name.startswith('input_'):
                inputs += 1
            else:
                assert data_name.startswith('output_'), 'only input_ and output_ prefixes are expected'
                outputs += 1
        
        print(f'    {{"{test_name}", {inputs}, {outputs}}},')
    print('};')


def dump_filter_switch_stub(check_fn = None):
    print('--- This content is a stub for test_onnx_conformance_layer_filter_<...>.inl.hpp ---')
    print('// Update note: execute <opencv_extra>/testdata/dnn/onnx/generate_conformance_list.py')
    print('BEGIN_SWITCH()')
    for test_name in os.listdir(NODE_DIR):
        print(f'CASE({test_name})')
        result = True
        if check_fn is not None:
            result = check_fn(test_name)
        if result:
            print(f'    // pass')
        else:
            print(f'    SKIP;')

    print('END_SWITCH()')


def dump_denylist_stub(check_fn = None):
    print('--- This content is a stub for test_onnx_conformance_layer_filter_<...>.inl.hpp ---')
    print('// Update note: execute <opencv_extra>/testdata/dnn/onnx/generate_conformance_list.py')
    for test_name in os.listdir(NODE_DIR):
        result = True
        if check_fn is not None:
            result = check_fn(test_name)
            if not result:
                print(f'"{test_name}",')
        else:
            print(f'//"{test_name}",')

    print('// end of list')


def parse_test_tags(fname_test_results):
    from xml.dom.minidom import parse
    log = parse(fname_test_results)

    test_results = {}
    for xmlnode in log.getElementsByTagName("testcase"):
        fixture = xmlnode.getAttribute("classname")
        if 'Test_ONNX_conformance' != fixture:
            continue
        name = xmlnode.getAttribute("name")
        if not name.startswith('Layer_Test/'):
            continue
        value_param = xmlnode.getAttribute("value_param")
        # (test_abs, OCV/CPU)
        if not value_param.endswith(', OCV/CPU)'):
            continue

        test_name = value_param[1:-len(', OCV/CPU)')]

        properties = {
            prop.getAttribute("name") : prop.getAttribute("value")
            for prop in xmlnode.getElementsByTagName("property")
            if prop.hasAttribute("name") and prop.hasAttribute("value")
        }

        tags = properties.get('tags', '').split(',')
        #print(test_name, tags)
        test_results[test_name] = tags

    assert len(test_results) > 0
    return test_results


def dump_parser_filter(fname_test_results):
    test_results = parse_test_tags(fname_test_results)

    def check_test_case_parser_error(test_name):
        if not test_name in test_results:
            return false
        tags = test_results[test_name]
        return not 'dnn_error_parser' in tags and not 'dnn_skip_parser' in tags

    with open(os.path.join(ROOT_DIR, 'stubs/test_onnx_conformance_layer_filter_parser.inl.hpp'), 'w') as f:
        with redirect_stdout(f):
            dump_denylist_stub(check_test_case_parser_error)
    

def main():
    with open(os.path.join(ROOT_DIR, 'stubs/test_list.txt'), 'w') as f:
        with redirect_stdout(f):
            dump_test_list()
    with open(os.path.join(ROOT_DIR, 'stubs/test_onnx_conformance_layer_filter_XYZ.inl.hpp'), 'w') as f:
        with redirect_stdout(f):
            dump_filter_switch_stub()
    if len(sys.argv) > 1:
        dump_parser_filter(sys.argv[1])
    print('Check for updates!')
    print('Execute in <opencv_extra>: git diff -- testdata/dnn/onnx/conformance/stubs')


if __name__ == '__main__':
    sys.exit(main())
