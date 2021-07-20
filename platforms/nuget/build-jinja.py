import os
from jinja2 import Environment, FileSystemLoader
PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)

params = {
'id': 'OpenCVNuget',
'version': '4.5.2',
'description': 'OpenCV Nuget Package for C++',
'tags': 'OpenCV, opencv',
'authors': 'OpenCV',
'compilers': ['vc14', 'vc15', 'vc16'],
'architectures': ['x86', 'x64'],
}

def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)
def create_index_html():
    fname = "build/opencv.nuspec"
    context = {
        'params': params
    }
    with open(fname, 'w') as f:
        html = render_template('OpenCVNuget.nuspec', context)
        f.write(html)
def main():
    create_index_html()
if __name__ == "__main__":
    main()