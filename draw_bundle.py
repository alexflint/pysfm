import numpy as np
from numpy import *
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages

def draw_predictions(bundle, i, *args, **kwargs):
    msm_pts = nonzero(bundle.msm_mask[i])[0]
    xs = array([ bundle.predict(i,j) for j in msm_pts ])
    plot(xs[:,0], xs[:,1], *args, **kwargs)

def draw_measurements(bundle, i, *args, **kwargs):
    xs = bundle.msm[i][ bundle.msm_mask[i] ]
    plot(xs[:,0], xs[:,1], *args, **kwargs)

def draw_views(bundle):
    matlen = ceil(sqrt(bundle.ncameras))
    for i in range(bundle.ncameras):
        subplot(matlen, matlen, i+1)
        draw_predictions(bundle, i, '.r')
        draw_measurements(bundle, i, 'xg')

def output_views(bundle, path):
    pdf = PdfPages(path)
    clf()
    draw_views(bundle)
    pdf.savefig()
    pdf.close()
