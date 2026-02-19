from docutils import nodes


def logo_role(name, rawtext, text, *args, **kwargs):
    node = nodes.inline(text=text if text != 'null' else '')
    node['classes'] += ['inline-logo', name]
    if text == 'null':
        node['classes'].append('empty')
    return [node], []


def setup(app):
    app.add_role('hybridrag', logo_role)
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
