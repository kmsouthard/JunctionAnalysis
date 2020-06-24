#!/usr/bin/env python
from __future__ import division
import skimage.io as io
io.use_plugin('tifffile')

from skimage import io, img_as_float
from skimage.filters import threshold_otsu, median, threshold_li
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import disk, square, remove_small_objects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier

from skimage.morphology import closing, dilation, opening, erosion
from skimage.segmentation import clear_border

from scipy.ndimage import median_filter 
from skimage import io, exposure, img_as_uint, img_as_float
from skimage.morphology import reconstruction
from skimage import exposure

def load_movie(filename):
    movie = io.imread(filename)
    return movie


def movie_summary(movie, channel, scaling_factor, filename, path):
    #load movie dimensions
    x_size, channels, z_size, y_size = movie.shape
    sm_time = x_size//scaling_factor
    
    nrows = np.int(np.ceil(np.sqrt(sm_time)))
    ncols = np.int(sm_time//nrows+1)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows)) 
    count = 0
    for n in range(0, x_size, scaling_factor):
        i = count // ncols
        j = count % ncols
        axes[i, j].imshow(movie[n,channel, ...], 
                        interpolation='nearest', 
                        cmap='gray')
        count += 1
    
    ## Remove empty plots 
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    #save figure
    plt.savefig(path+'summary'+str(channel)+filename+'.png', format='png')
    plt.close()
    return

def thres_movie(movie, channel, threshold_meth):
    max_int_proj = movie.max(axis=0)
    thresh_global = threshold_otsu(max_int_proj[channel])
    return thresh_global
    
def thres_movie_median(movie, channel, threshold_meth):
    med_int_proj = np.median(movie[:, channel, :,:])
    thresh_global = threshold_otsu(med_int_proj)
    return thresh_global
    
def subtract_background(movie, radius, height, channel):
    background_subtracted = movie.copy()
    
    for x, frame in enumerate(movie[:, channel, :, :]):
        smoothed = median_filter(frame, radius)
        seed = np.copy(smoothed) - (height*movie[:,0,...].max())
        mask = smoothed
        dilated = reconstruction(seed, mask, method='dilation')
        hdome = smoothed - dilated
        background_subtracted[x,channel] = hdome 
        
    return background_subtracted

def thres_movie_perFrame(movie, channel, threshold_meth, min_thresh, max_thresh):

    thresh_frame = []

    ## thresholding for each frame in x_size series:
    for x, frame in enumerate(movie[:, channel, :, :]):
        thresh = threshold_meth(frame)
        if thresh < min_thresh:
            thresh_frame.append(min_thresh)
        if thresh > max_thresh:
            thresh_frame.append(max_thresh)
        else:
            thresh_frame.append(thresh)
    
    return thresh_frame     
     
            
def smooth_movie(movie, radius):
    smoothed_stack = movie.copy()

    for z, frame in enumerate(movie[:, 0, :, :]):
        smoothed = median_filter(frame, size = (radius,radius))
        smoothed_stack[z, 0] = smoothed
        
    for z, frame in enumerate(movie[:, 1, :, :]):
        smoothed = median_filter(frame, size = (radius,radius))
        smoothed_stack[z, 1] = smoothed
        
    
    return smoothed_stack


def label_movie(movie, channel, threshold, segmentation_meth, shape, size, min_radius):
    labeled_stack = np.zeros_like(movie[:,channel,:,:])
    for z, frame in enumerate(movie[:, channel, :, :]):
        smoothed = median(frame, shape(size))
        im_max = smoothed.max()
        if im_max < threshold:
            labeled_stack[z] = np.zeros(smoothed.shape, dtype=np.int32)
        else:
            bw = segmentation_meth(smoothed > threshold, shape(size))
            cleared = bw.copy()
            #cleared = clear_border(cleared)
            cleared = remove_small_objects(cleared, min_radius)
            labeled_stack[z] = label(cleared)
    return labeled_stack
    
def label_movie_frame(movie, channel, threshold, segmentation_meth, shape, size, min_radius):
    labeled_stack = np.zeros_like(movie[:,channel,:,:])

    ## Labeling for each frame in x_size series:
    for z, frame in enumerate(movie[:,channel,:,:]):
        im_max = frame.max()
        if im_max < threshold[z]:
            labeled_stack[z] = np.zeros_like(movie[z,channel,:,:])
        else:
            bw = segmentation_meth(frame > threshold[z], shape(size))
            cleared = bw.copy()
            #clear_border(cleared)
            cleared = remove_small_objects(cleared, min_radius)
            labeled_stack[z] = label(cleared)
    return labeled_stack

def label_movie_box(movie, channel, line_start, line_end):
    labeled_stack = np.zeros_like(movie[:,channel,:,:])

    ## Labeling for each frame in x_size series:
    for z, frame in enumerate(movie[:,channel,:,:]):
        line = np.zeros_like(frame)
        line[:, line_start:line_end] = 1
        labeled_stack[z] = label(line)
    return labeled_stack

def label_movie_boarder(movie, channel, threshold, segmentation_meth, shape, size):
    labeled_stack = np.zeros_like(movie[:,channel,:,:])
    for z, frame in enumerate(movie[:, channel, :, :]):
        im_max = frame.max()
        if im_max < threshold:
            labeled_stack[z] = np.zeros_like(movie[z,channel,:,:])
        else:
            bw = segmentation_meth(frame > threshold, shape(size))
            cleared = bw.copy()
            #clear_border(cleared)
            labeled_stack[z] = label(cleared)
    return labeled_stack
    

def segmentation_summary(movie, channel, smoothed_stack, labeled_stack, scaling_factor, filename, path):
    x_size, channels, z_size, y_size= movie.shape
    sm_time = x_size//scaling_factor
    
    nrows = np.int(np.ceil(np.sqrt(sm_time)))
    ncols = np.int(sm_time//nrows+1)
        
    fig, axes = plt.subplots(nrows, ncols*2, figsize=(3*ncols, 1.5*nrows))
    count = 0
    for n in range(0, x_size, scaling_factor):
        i = count // ncols
        j = count % ncols * 2
        count += 1
        axes[i, j].imshow(movie[n, channel, ...], interpolation='nearest', cmap='gray')
        axes[i, j+1].imshow(labeled_stack[n, ...], interpolation='nearest', cmap='Dark2')
        
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        axes[i, j+1].set_xticks([])
        axes[i, j+1].set_yticks([])
    
    ## Remove empty plots 
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    
            
    fig.tight_layout()
    #save image!!!!
    plt.savefig(path+'seg'+str(channel)+filename+'.png', format='png')
    plt.close()
    return


def measure_properties(movie, channel, labeled_stack, background):
    properties = []
    columns = ('x', 'y', 'z', 'I', 'A', 'I*A', 'radius')
    indices = []
    for z, frame in enumerate(labeled_stack):
        f_prop = regionprops(frame.astype(np.int), intensity_image = movie[z, channel])
        for d in f_prop:
            radius = (d.area/np.pi)**0.5
            properties.append([d.weighted_centroid[0],
                            d.weighted_centroid[1],
                            z, np.clip(d.mean_intensity-background, 0, None), d.area,
                            np.clip(d.mean_intensity-background, 0, None) * d.area,
                            radius])
            indices.append(d.label)
    if not len(indices):
        all_props = pd.DataFrame([], index=[])
    indices = pd.Index(indices, name='label')
    properties = pd.DataFrame(properties, index=indices, columns=columns)
    #properties['I'] /= properties['I'].max()
    return properties

def labeled_summary(movie,channel, scaling_factor,properties, labeled_stack, filename, path):
    x_size, channels, z_size, y_size= movie.shape
    sm_time = x_size//scaling_factor
    
    nrows = np.int(np.ceil(np.sqrt(sm_time)))
    ncols = np.int(sm_time//nrows+1)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    
    count = 0
    for n in range(0, x_size, scaling_factor):
        plane_props = properties[properties['x'] == n]
        if not(plane_props.shape[0]):
            continue
        i = count // ncols
        j = count % ncols
        count += 1
        axes[i, j].imshow(labeled_stack[n, ...],
                        interpolation='bicubic', cmap='Dark2')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        x_lim = axes[i, j].get_xlim()
        y_lim = axes[i, j].get_ylim()    
        
        
        axes[i, j].scatter(plane_props['y'], plane_props['x'],  
                        s=plane_props['I']*200, alpha=0.4)
        axes[i, j].scatter(plane_props['y'], plane_props['x'],
                        s=40, marker='+', alpha=0.4)
        axes[i, j].set_xlim(x_lim)
        axes[i, j].set_ylim(y_lim)       
    
    ## Remove empty plots 
    for ax in axes.ravel():
        if not(len(ax.images)):
                fig.delaxes(ax)  
            
    #fig.tight_layout()
    #figure save!!!!
    plt.savefig(path+'segAll'+str(channel)+filename+'.png', format='png')
    plt.close()
    return

def cluster_points(properties, max_dist):
    positions = properties[['x', 'y']].copy()

    dist_mat = dist.squareform(dist.pdist(positions.values))
    link_mat = hier.linkage(dist_mat)
    cluster_idx = hier.fcluster(link_mat, max_dist,
                                criterion='distance')
    properties['new_label'] = cluster_idx
    properties.set_index('new_label', drop=True, append=False, inplace=True)
    properties.index.name = 'label'
    properties = properties.sort_index()
    return properties

def df_average(df, weights_column):
    '''Computes the average on each columns of a dataframe, weighted
    by the values of the column `weight_columns`.
    
    Parameters:
    -----------
    df: a pandas DataFrame instance
    weights_column: a string, the column name of the weights column 
    
    Returns:
    --------
    
    values: pandas DataFrame instance with the same column names as `df`
        with the weighted average value of the column
    '''
    
    values = df.copy().iloc[0]
    norm = df[weights_column].sum()
    for col in df.columns:
        try:
            v = (df[col] * df[weights_column]).sum() / norm
        except TypeError:
            v = df[col].iloc[0]
        values[col] = v
    return values

def cluster_positions(properties):
    cluster_positions = properties.groupby(level='label').apply(df_average, 'I')
    return cluster_positions


def cluster_plot(movie,channel, cluster_positions, properties, filename, path, xy_scale, z_scale):
    x_size, channels, z_size, y_size= movie.shape

    labels = cluster_positions.index.tolist()

    fig = plt.figure(figsize=(12, 12))
    colors = plt.cm.jet(properties.index.astype(np.int32))
    
    # xy projection:
    ax_xy = fig.add_subplot(111)
    ax_xy.imshow(movie.max(axis=0)[channel], cmap='gray')
    ax_xy.scatter(properties['y'],
                properties['x'],
                c=colors, alpha=0.2)
    
    
    ax_xy.scatter(cluster_positions['y'],
                cluster_positions['x'],
                c='r', s=50, alpha=1.)
    
    
    for i, txt in enumerate(labels):
        ax_xy.annotate(txt, (cluster_positions.iloc[i]['y'], cluster_positions.iloc[i]['x']))
    
    divider = make_axes_locatable(ax_xy)
    ax_yz = divider.append_axes("top", 2, pad=0.2, sharex=ax_xy)
    ax_yz.imshow(movie.max(axis=1)[:,channel], aspect=z_scale/xy_scale, cmap='gray')
    ax_yz.scatter(properties['y'],
                properties['z'],
                c=colors, alpha=0.2)
    
    ax_yz.scatter(cluster_positions['y'],
                cluster_positions['z'],
                c='r', s=50, alpha=1.)
    
    
    ax_zx = divider.append_axes("right", 2, pad=0.2, sharey=ax_xy)
    ax_zx.imshow(movie.max(axis=2)[:,channel], aspect=xy_scale/z_scale, cmap='gray')
    ax_zx.scatter(properties['z'],
                properties['x'],
                c=colors, alpha=0.2)
    
    ax_zx.scatter(cluster_positions['z'],
                cluster_positions['x'],
                c='r', s=50, alpha=1.)
    
    plt.draw()
    plt.savefig(path+'clust'+str(channel)+filename+'.png', format='png')
    plt.close()
    return

##testing code
#smooth_size = 1 # pixels
#min_radius = 2
#max_radius = 30
#t_scale = 1.0 #frame per second
#xy_scale = 0.106 #um per pixel
#
#
#path = fill
#filename = fill
#movie = fill
#movie_summary(movie, 55, filename, path)
#threshold = thres_movie(movie, threshold_otsu)
##smoothed_movie = smooth_movie(movie, smooth_size, median, disk)
#labeled_movie = label_movie(movie, threshold, opening, disk, 1)
#segmentation_summary(movie, movie, labeled_movie, 11, filename, path)
#properties = measure_properties(movie, labeled_movie)
#labeled_summary(movie, 11,properties, labeled_movie, filename, path)
#properties_clust = cluster_points(properties, 25)
#cluster_pos = cluster_positions(properties_clust)
#cluster_plot(movie, cluster_pos, properties_clust, filename, path, xy_scale)