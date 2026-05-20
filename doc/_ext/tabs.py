from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective


class DivNode(nodes.General, nodes.Element):
    pass


class DivDirective(SphinxDirective):
    has_content = True
    optional_arguments = 1
    option_spec = {"class": directives.unchanged}

    def run(self):
        node = DivNode()
        css = self.arguments[0] if self.arguments else self.options.get("class", "")
        node["css_class"] = css
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def _visit_div(self, node):
    self.body.append(f'<div class="{node["css_class"]}">')


def _depart_div(self, node):
    self.body.append("</div>")


class TabSetNode(nodes.General, nodes.Element):
    pass


class TabItemNode(nodes.General, nodes.Element):
    pass


class TabSetDirective(SphinxDirective):
    has_content = True
    option_spec = {"class": directives.unchanged}

    def run(self):
        node = TabSetNode()
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class TabItemDirective(SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    option_spec = {"sync": directives.unchanged, "class": directives.unchanged}

    def run(self):
        node = TabItemNode()
        node["label"] = self.arguments[0]
        node["sync"] = self.options.get("sync", "")
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def _visit_tab_set(self, node):
    items = [n for n in node.children if isinstance(n, TabItemNode)]
    labels_html = '<div class="ocv-tab-labels" role="tablist">'
    for i, item in enumerate(items):
        active = " ocv-tab-active" if i == 0 else ""
        sync = item.get("sync", "")
        labels_html += (
            f'<button class="ocv-tab-btn{active}" role="tab"'
            f' data-sync="{sync}" onclick="ocvTabClick(this)">'
            f'{item["label"]}</button>'
        )
    labels_html += "</div>"
    self.body.append(f'<div class="ocv-tabset">{labels_html}')


def _depart_tab_set(self, node):
    self.body.append("</div>")


def _visit_tab_item(self, node):
    parent = node.parent
    items = [n for n in parent.children if isinstance(n, TabItemNode)]
    idx = items.index(node)
    sync = node.get("sync", "")
    active = " ocv-tab-active" if idx == 0 else ""
    self.body.append(
        f'<div class="ocv-tab-panel{active}" role="tabpanel" data-sync="{sync}">'
    )


def _depart_tab_item(self, node):
    self.body.append("</div>")


def setup(app):
    app.add_node(DivNode, html=(_visit_div, _depart_div))
    app.add_node(TabSetNode, html=(_visit_tab_set, _depart_tab_set))
    app.add_node(TabItemNode, html=(_visit_tab_item, _depart_tab_item))
    app.add_directive("div", DivDirective)
    app.add_directive("tab-set", TabSetDirective)
    app.add_directive("tab-item", TabItemDirective)
    return {"version": "0.1", "parallel_read_safe": True}
