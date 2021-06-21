'''
Extension providing the new directive ``code-output``

``.. code-output::`` will create a code output cell that looks exactly like
the output cells created with the sphinx-gallery extension, but without
having to write *Python* code that is executed.

Based on

- https://www.sphinx-doc.org/en/master/development/tutorials/todo.html
- https://stackoverflow.com/q/51305347
'''

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class CodeOutputNode(nodes.Element):
    pass

def visit_code_output_node(self, node):
    self.body.extend([
        '<p class="sphx-glr-script-out">Out:</p>',
        '<div class="sphx-glr-script-out highlight-none notranslate">',
        '<div class="highlight">',
        '<pre>'
    ])
    self.body.append('<span></span>')
    for i, line in enumerate(node.rawsource):
        if i > 0: self.body.append('\n')
        self.body.append(line)

def depart_code_output_node(self, node):
    self.body.append('</pre>')
    self.body.append('</div>')
    self.body.append('</div>')

def setup(app):
    app.add_node(CodeOutputNode,
                 html=(visit_code_output_node, depart_code_output_node))
    app.add_directive('code-output', CodeOutputDirective)
    return {
        'version': 0.1,
        'parallel_read_safe': True,
        'parallel_write_safe': True
    }


class CodeOutputDirective(SphinxDirective):

    # this enables content in the directive
    has_content = True

    def run(self):
        node = CodeOutputNode(self.content)
        return [node]
