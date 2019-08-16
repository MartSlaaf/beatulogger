from IPython.core.display import HTML


def get_color_code(value, coloring_type):
    if coloring_type == 'positive':
        r = 0
        g = value * 255
        b = 0
    elif coloring_type == 'positive-negative':
        r = (1 - value) * 255
        g = value * 255
        b = 0
    elif coloring_type == 'cathegory':
        raise NotImplementedError('colorisation with cathegory id is not implemented yet!')
    else:
        raise Exception('Unknown coloring type "{coloring_type}"')
    return r, g, b


from cgi import escape

def decorate_text_with_words(text, intensity, inverse_dictionary=None, coloring_type='positive'):

    if len(text) != len(intensity):
        raise ValueError('Non-coherent lengths of text and intensities')

    if inverse_dictionary is not None:
        text = [inverse_dictionary[w] for w in text]

    html_result = ''
    for t, i in zip(text, intensity):
        r, g, b = get_color_code(i, coloring_type)
        t = escape(t)
        html_result += f'<span style="background-color: rgb({r}, {g}, {b}">{t} </span>'
    fin_html = ''.join(seq_diane)
    display(HTML(fin_html))
