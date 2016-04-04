from matplotlib import rc
import matplotlib.font_manager as font_manager

import seaborn as sns

def setup():

    # path = '/usr/share/fonts/linux-libertine/LinBiolinum_R.otf'
    # prop = font_manager.FontProperties(fname=path)
    sns.set(context='paper')
    rc('text', usetex=True)
    rc('text', **{'latex.preamble': r'\usepackage{libertine},\usepackage[libertine]{newtxmath},\renewcommand*\familydefault{\sfdefault},\usepackage[T1]{fontenc}'})
