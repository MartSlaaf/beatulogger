from IPython.core.display import HTML


def get_color_code(value, coloring_type):
    if coloring_type == 'positive':
        r = (1 - value) * 255
        g = 255
        b = (1 - value) * 255
    elif coloring_type == 'positive-negative':
        r = (1 - value) * 255
        g = value * 255
        b = 255 * (0.5 - abs(value - 0.5))
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
        html_result += f'<span style="background-color: rgb({r}, {g}, {b})">{t} </span>'
    display(HTML(html_result))
    return html_result


import os
import tempfile
import numpy as np

def make_rotation_gif(ax, fig, file_to_save, elevation=30):
    tmpdir = tempfile.TemporaryDirectory()
    for azim in np.arange(0, 355, 5):
        ax.view_init(elev=elevation, azim=azim)
        fig.savefig(os.path.join(tmpdir.name, f'tmp_a{azim:03}.png'))
    pathes_of_pngs = os.path.join(tmpdir.name, "tmp_a*.png")
    os.system(f'convert -loop 0 -delay 10 {pathes_of_pngs} {file_to_save}')
    tmpdir.cleanup()
