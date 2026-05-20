import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("_ext"))

project = "OpenCV"
author = "OpenCV Team"
copyright = f"{datetime.now().year}, OpenCV Team"
release = "5.x"

extensions = [
    "myst_parser",
    "tabs",
    "doxysnippet",
    "opencv_code_links",
    "breathe",
    "exhale",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 4
myst_dmath_double_inline = True

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap",
    "custom.css",
]
html_js_files = ["opencv-code-links.js", "copybutton.js", "theme-toggle.js"]
html_title = "OpenCV Documentation"
html_logo = "_static/opencv-logo-white.png"
html_favicon = "_static/opencv.ico"
html_show_sourcelink = False
html_copy_source = False
html_meta = {"opencv-code-links": "enable"}

html_theme_options = {
    "navbar_start": ["navbar-logo", "version-badge"],
    "navbar_center": [],
    "navbar_end": ["external-nav", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 7,
    "show_prev_next": True,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "use_edit_page_button": False,
    "external_links": [
        {"name": "Main Page", "url": "https://docs.opencv.org/5.x/index.html"},
        {"name": "Related Pages", "url": "https://docs.opencv.org/5.x/pages.html"},
        {"name": "Namespaces", "url": "https://docs.opencv.org/5.x/namespaces.html"},
        {"name": "Classes", "url": "https://docs.opencv.org/5.x/annotated.html"},
        {"name": "Files", "url": "https://docs.opencv.org/5.x/files.html"},
        {"name": "Examples", "url": "https://docs.opencv.org/5.x/examples.html"},
        {"name": "Java documentation", "url": "https://docs.opencv.org/5.x/javadoc/index.html"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/opencv/opencv",
            "icon": "fab fa-github",
        },
    ],
    "logo": {
        "image_light": "_static/opencv-logo.png",
        "image_dark": "_static/opencv-logo-white.png",
        "alt_text": "OpenCV — Open Source Computer Vision Library",
    },
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
    "index": [],
    "introduction": [],
    "faq": [],
}

suppress_warnings = ["myst.header"]

breathe_projects = {
    "OpenCV": os.path.join(os.path.dirname(__file__), "_build/doxygen/xml"),
}
breathe_default_project = "OpenCV"
breathe_domain_by_extension = {"hpp": "cpp", "h": "cpp"}

exhale_args = {
    "containmentFolder":    "./api",
    "rootFileName":         "library_root.rst",
    "rootFileTitle":        "C++ API Reference",
    "doxygenStripFromPath": os.path.join(os.path.dirname(__file__), ".."),
    "createTreeView":       True,
    "exhaleExecutesDoxygen": False,
}

def setup(app):
    # exhale 0.3.x skips os.makedirs before writing RST files; patch to fix first-run failure
    import exhale.graph as _eg
    _orig = _eg.ExhaleRoot.generateSingleNodeRST

    def _patched(self, node):
        os.makedirs(os.path.dirname(node.file_name), exist_ok=True)
        return _orig(self, node)

    _eg.ExhaleRoot.generateSingleNodeRST = _patched

    # breathe 4.36.x cpp_classes missing "property" kind from Doxygen XML; map to var
    from breathe.renderer.sphinxrenderer import DomainDirectiveFactory, CPPMemberObject
    DomainDirectiveFactory.cpp_classes.setdefault("property", (CPPMemberObject, "var"))
