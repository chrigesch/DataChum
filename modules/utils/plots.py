# Import the required libraries

import base64
from io import BytesIO

# https://stackoverflow.com/questions/48717794/matplotlib-embed-figures-in-auto-generated-html
def _convert_fig_to_html(fig):
    """ Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding. """
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    return '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
