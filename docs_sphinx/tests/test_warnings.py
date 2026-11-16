# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Run with the docs requirements: python -m unittest discover -s docs_sphinx/tests."""
import io
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from conf_helpers import targets, translate, xml_render


class WarningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.root_patch = patch.object(targets, "SPHINX_INPUT_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        targets._label_owners.cache_clear()
        self.addCleanup(targets._label_owners.cache_clear)

    def write(self, docname, text):
        path = self.root / (docname + ".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_group_owns_repeated_namespace_label(self):
        text = "(Member_ID)=\n### Member\n"
        self.write("main_modules/group", text)
        self.write("extra_modules/namespace_cv", text)
        self.assertEqual(targets.localize_api_labels(text, "main_modules/group"), text)
        duplicate = targets.localize_api_labels(text, "extra_modules/namespace_cv")
        self.assertIn('<a id="member-id"></a>', duplicate)
        self.assertNotIn("(Member_ID)=", duplicate)

    def test_label_examples_do_not_claim_targets(self):
        text = "```text\n(member)=\n```\n"
        self.write("main_modules/a_example", text)
        self.write("main_modules/z_actual", "(member)=\n# Member\n")
        self.assertEqual(targets.localize_api_labels(text, "main_modules/a_example"), text)
        self.assertEqual(targets._label_owners()["member"], "main_modules/z_actual")

    def test_note_body_is_separated_from_options(self):
        text = translate._translate("@note : This is note text.\n", "tutorials/test")
        self.assertIn(":::{note}\n\n: This is note text.", text)
        # Exercise the XML block renderer with a note nested in a paragraph.
        node = ET.fromstring('<detaileddescription><para><simplesect kind="note">'
                             '<para>: Note text.</para></simplesect></para></detaileddescription>')
        text = xml_render._doxygen_desc_to_md(node, 2)
        self.assertIn(":::{note}\n\n: Note text.", text)

    def test_subpages_keep_their_api_tree(self):
        self.write("main_modules/cudaarithm", "# CUDA arithmetic\n")
        self.write("extra_modules/cudaarithm", "# CUDA arithmetic\n")
        with patch.object(translate, "SPHINX_INPUT_ROOT", self.root), \
                patch.dict(translate._ANCHOR_TO_DOC,
                           {"api_cudaarithm": "extra_modules/cudaarithm"}):
            text = translate._translate("- @subpage api_cudaarithm\n", "main_modules/cuda")
        self.assertIn("/main_modules/cudaarithm", text)
        self.assertNotIn("/extra_modules/cudaarithm", text)

    def test_sphinx_keeps_global_and_page_local_links(self):
        from sphinx.application import Sphinx
        self.write("index", "# API\n\n```{toctree}\n\nmain_modules/group\n"
                   "extra_modules/namespace_cv\n```\n")
        for docname in ("main_modules/group", "extra_modules/namespace_cv"):
            self.write(docname, "# API view\n\n(Member_ID)=\n## Member\n\n"
                       '[Local member](#Member_ID)\n')
        (self.root / "conf.py").write_text(
            "extensions = ['myst_parser']\nmaster_doc = 'index'\n"
            "source_suffix = {'.md': 'markdown'}\n", encoding="utf-8")
        warnings = io.StringIO()
        app = Sphinx(str(self.root), str(self.root), str(self.root / "_html"),
                     str(self.root / "_doctrees"), "html", status=io.StringIO(),
                     warning=warnings, freshenv=True)
        def source_read(app, docname, source):
            source[0] = targets.localize_api_labels(source[0], docname)
        app.connect("source-read", source_read)
        app.build(force_all=True)
        self.assertEqual(warnings.getvalue(), "")
        self.assertEqual(app.env.domains["std"].labels["member_id"][0], "main_modules/group")
        html = (self.root / "_html/extra_modules/namespace_cv.html").read_text()
        self.assertIn('id="member-id"', html)
        self.assertIn('href="#extra-modules-namespace-cv-member-id"', html)
        self.assertIn('id="extra-modules-namespace-cv-member-id"', html)


if __name__ == "__main__":
    unittest.main()
