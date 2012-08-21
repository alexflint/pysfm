import numpy as np
from numpy import *
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages

def draw_points(xs, *args, **kwargs):
    xs = np.asarray(xs)
    if xs.size > 0:
        plot(xs[:,0], xs[:,1], *args, **kwargs)

def draw_predictions(bundle, i, *args, **kwargs):
    xs = [ bundle.predict(i,j) for j,t in enumerate(bundle.tracks) if t.has_measurement(i) ]
    draw_points(xs, *args, **kwargs)

def draw_measurements(bundle, i, *args, **kwargs):
    xs = [ t.get_measurement(i) for t in bundle.tracks if t.has_measurement(i) ]
    draw_points(xs, *args, **kwargs)

def draw_views(bundle, inds):
    matlen = ceil(sqrt(len(bundle.cameras)))
    for i in range(len(bundle.cameras)):
        subplot(matlen, matlen, i+1)
        draw_predictions(bundle, i, '.r')
        draw_measurements(bundle, i, 'xg')
        labels = [ 'Projections', 'Measurements' ]

        # Draw outliers if they are known
        if hasattr(bundle, 'outlier_mask'):
            js = nonzero(bundle.outlier_mask[i])[0]
            outlier_msms = [ bundle.measurement(i,j) for j in js ]
            draw_points(outlier_msms, 'xm')
            labels.append('Outliers (ground truth)')

        #legend(labels, textsize='xx-small')
        xticks([])
        yticks([])
            

def output_views(bundle, path):
    pdf = PdfPages(path)
    clf()
    draw_views(bundle)
    pdf.savefig()
    pdf.close()
