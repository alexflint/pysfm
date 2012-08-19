import numpy as np
from numpy import *
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages

def draw_points(xs, *args, **kwargs):
    plot(xs[:,0], xs[:,1], *args, **kwargs)

def draw_predictions(bundle, i, *args, **kwargs):
    msm_pts = nonzero(bundle.msm_mask[i])[0]
    xs = array([ bundle.predict(i,j) for j in msm_pts ])
    draw_points(xs, *args, **kwargs)

def draw_measurements(bundle, i, *args, **kwargs):
    xs = bundle.msm[i][ bundle.msm_mask[i] ]
    draw_points(xs, *args, **kwargs)

def draw_views(bundle):
    matlen = ceil(sqrt(bundle.ncameras))
    for i in range(bundle.ncameras):
        subplot(matlen, matlen, i+1)
        draw_predictions(bundle, i, '.r')
        draw_measurements(bundle, i, 'xg')
        labels = [ 'Projections', 'Measurements' ]

        # Draw outliers if they are known
        if hasattr(bundle, 'outlier_mask'):
            outlier_msms = bundle.msm[ i, bundle.outlier_mask[i] ]
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
