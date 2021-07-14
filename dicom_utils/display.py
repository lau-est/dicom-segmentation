from dicom_utils import reader
import pydicom as dcm
import numpy as np
import scipy.ndimage
from skimage import measure
from skimage import morphology
from matplotlib import pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.graph_objs import *

def plot_slices(slices, wtype='lung', nrows=5, ncols=5, start=10, step=5):
    fig, ax = plt.subplots(nrows, ncols, figsize=[8,8])

    counter = 0

    for i in range(nrows):
        for j in range(ncols):
            index = counter * step + start
            
            if index > len(slices):
                break

            ax[i, j].set_title("slice %d" % index)
            ax[i, j].imshow(reader.get_windowed_image(slices[index], wtype), cmap='gray')
            ax[i, j].axis('off')
            counter += 1
    plt.show()


def resample_ct(slices, new_spacing=[1,1,1]):
    ct_shape = [len(slices), slices[0].pixel_array.shape[0], slices[0].pixel_array.shape[1]]
    spacing  = [slices[0].SliceThickness, slices[0].PixelSpacing[0] , slices[0].PixelSpacing[1]]
    spacing  = np.array(list(spacing))

    resize_factor      = spacing  / new_spacing
    new_real_shape     = ct_shape * resize_factor
    new_shape          = np.round(new_real_shape)
    real_resize_factor = new_shape / ct_shape
    new_spacing        = spacing   / real_resize_factor

    images  = np.zeros(ct_shape, dtype = np.int16)
    counter = 0

    for slc in slices:
        images[counter, :, :] = reader.get_image_hu(slc)
        counter += 1

    images = scipy.ndimage.interpolation.zoom(images, real_resize_factor)

    return images, new_spacing


def make_mesh(images, threshold=-300, step_size=1):
    p                       = images.transpose(2, 1, 0)
    #verts, faces, norm, val = measure.marching_cubes_classic() (p, threshold, step_size=step_size, allow_degenerate=True) 
    verts, faces, _, _ = measure.marching_cubes_lewiner(p)

    return p, verts, faces


def plotly_3d(verts, faces):
    x, y, z  = zip(*verts)
    colormap = ['rgb(220, 220, 220)', 'rgb(220, 220, 220)']

    fig      = FF.create_trisurf(
        x          = x                     ,
        y          = y                     ,
        z          = z                     ,
        plot_edges = False                 ,
        colormap   = colormap              ,
        simplices  = faces                 ,
        backgroundcolor = 'rgb(64, 64, 64)',
        title           = '3D CT Scan'
    )

    iplot(fig)


def plt_3d(p, verts, faces):
    fig        = plt.figure(figsize=(10, 10))
    ax         = fig.add_subplot(111, projection='3d')

    mesh       = Poly3DCollection(verts[faces], alpha=0.70)#(verts[faces], linewidths=0.05, alpha=1)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    #ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()


def plot_3d(slices, threshold = 500):
    print("Resampling CT Scan ...")
    t1, _ = resample_ct(slices)
    print("Rendering 3D object...")
    p, v, f  = make_mesh(t1, threshold)
    plt_3d(p, v, f)


def plot(img, title=None):
    plt.imshow(img, cmap = 'gray')
    plt.title(title)
    plt.colorbar()
    plt.show()