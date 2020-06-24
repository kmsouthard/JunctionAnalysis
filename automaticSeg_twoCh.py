import os
import fnmatch
from segmentationFunctions_Ecad import *

#path to resliced images folder
path = "your path here"

#add together average intensity from each side and divide
#draw a line on each side (mask) and take a measure of each cell's intensity within the notch segementation
#simular to using Ecad to subtract

pattern = '*.tif'
smooth_size = 1 # pixels
#min_radius = 10
#max_radius = 1000
z_scale = 0.211 
xy_scale = 0.211 #um per pixel
summary = []
titles = ('filename', 'area_junction', 'area_out', 'int_junction', 'int_out', 'ratio', 'ratio_avg')
indx = []

i = 0
for (path, dirs, files) in os.walk(path):
    for filename in fnmatch.filter(files, pattern):
        print(filename)
        movie = load_movie(os.path.join(path, filename))

        smoothed_movie= smooth_movie(movie, smooth_size)
        
        #maximum threshold as the 99th percentile of the Notch channel intensity
        max_thresh = np.percentile(smoothed_movie[:,1, ...], 99)
        
        #minimum threshold as 80 percent of the Notch channel intensity
        min_thresh = np.percentile(smoothed_movie[:,1, ...],80)
        
        #threshold Ecad Junctions
        threshold_Ecad = thres_movie(smoothed_movie, 0, threshold_otsu)
        
        #threshold Notch intensity at membrane
        threshold_Notch = thres_movie_perFrame(smoothed_movie, 1, threshold_otsu, min_thresh, max_thresh)
        
        #segment movie with threshold
        labeled_movie_Ecad = label_movie(smoothed_movie, 0, threshold_Ecad, opening, disk, 1, min_radius = 20)
        labeled_movie_Notch = label_movie_frame(smoothed_movie, 1, threshold_Notch, closing, disk, 1, min_radius = 5)
        
        #subtract Ecad segmentation from Notch
        labeled_movie_Notch = labeled_movie_Notch*(np.logical_not(labeled_movie_Ecad).astype(int))
        
        #take a line from each side of image to get expression levels
        #define boxes
        labeled_movie_box1 = label_movie_box(smoothed_movie, 1,10,15)
        labeled_movie_box2 = label_movie_box(smoothed_movie, 1,-15,-10)
        #limit to regions overlaping with Notch segmentation
        notch_expression_1 = labeled_movie_Notch*(np.logical_and(labeled_movie_Notch, labeled_movie_box1).astype(int))
        notch_expression_2 = labeled_movie_Notch*(np.logical_and(labeled_movie_Notch, labeled_movie_box2).astype(int))
        
        
        #create summary figures for manual evaluation of segmentation (comment out to increase analsysis speed)
        segmentation_summary(smoothed_movie, 0, smoothed_movie, labeled_movie_Ecad, 2, 'ecad'+filename, path)
        segmentation_summary(smoothed_movie, 1, smoothed_movie, labeled_movie_Notch, 2, 'notch'+filename, path)
        segmentation_summary(smoothed_movie, 1, smoothed_movie, notch_expression_1, 2, 'cell_1'+filename, path)
        segmentation_summary(smoothed_movie, 1, smoothed_movie, notch_expression_2, 2, 'cell_2'+filename, path)
        
        #define Notch background as the minimum threshold
        N_series = pd.Series(threshold_Notch)
        background = N_series.min()
        movie_min = smoothed_movie[:,1,...].min()
        
        #measure Notch intensities
        properties_junction = measure_properties(smoothed_movie,1, labeled_movie_Ecad, background)
        properties_out = measure_properties(smoothed_movie,1, labeled_movie_Notch, background)
        #cell1
        properties_expression1 = measure_properties(smoothed_movie, 1, notch_expression_1, background)
        #cell2
        properties_expression2 = measure_properties(smoothed_movie, 1, notch_expression_2, background)
        
        #measure cadherin intensities
        prop_cad_exp1 = measure_properties(smoothed_movie, 0, notch_expression_1, smoothed_movie[:,0,...].min())
        prop_cad_exp2 = measure_properties(smoothed_movie, 0, notch_expression_2, smoothed_movie[:,0,...].min())
        prop_cad_junc = measure_properties(smoothed_movie, 0, labeled_movie_Ecad, smoothed_movie[:,0,...].min())
        
        #save properties
        properties_junction.to_csv(path+filename+'junction.csv')
        properties_out.to_csv(path+filename+'out.csv')
        
        properties_expression1.to_csv(path+filename+'expression1.csv')
        properties_expression2.to_csv(path+filename+'expression2.csv')
        
        #remove too large segments (may not be needed)
        #properties_junction = properties_junction[(properties_junction["A"] < 5000)]
        #properties_out = properties_out[(properties_out["A"] < 5000)]
        
        #total area/intensity junction
        total_area_junction = properties_junction['A'].sum()
        total_int_junction = properties_junction['I*A'].sum()
        
        #average of junction intensity
        if total_area_junction == 0:
            avg_int_junction = "NaN"
        else:
            avg_int_junction = total_int_junction/total_area_junction
            
        #total area/intensity membrane
        total_area_out = properties_out['A'].sum()
        total_int_out = properties_out['I*A'].sum()
        
        #average of membrane intensity
        if total_area_out == 0:
            avg_ing_junction = "NaN"
        else:
            avg_int_out = total_int_out/total_area_out
            
        #calculate intensity for each cell
        total_area_exp1 = properties_expression1['A'].sum()
        total_int_exp1 = properties_expression1['I*A'].sum()
        total_area_exp2 = properties_expression2['A'].sum()
        total_int_exp2 = properties_expression2['I*A'].sum()
        
        if  total_area_exp1 == 0 or total_area_exp2 == 0:
            ratio = "NaN"
            
        #remove cells with weak junctions     
        elif prop_cad_junc['I'].mean() < 2*(prop_cad_exp1['I'].mean() + prop_cad_exp2['I'].mean()):
            ratio = "NaN"
        
        else:
            avg_int_exp1 = total_int_exp1/total_area_exp1
            avg_int_exp2 = total_int_exp2/total_area_exp2
            
            ratio = avg_int_junction/(avg_int_exp1 + avg_int_exp2)
        #averaged
        ratio_avg = properties_junction['I'].mean()/(properties_expression1['I'].mean()+properties_expression2['I'].mean())
        # add to dataframe
        summary.append([filename, total_area_junction, total_area_out, avg_int_junction, avg_int_out, ratio, ratio_avg])
        i += 1
        #if i >= 1:
         #   break
            
summary = pd.DataFrame(summary, columns=titles)

#name output file
summary.to_csv(path+'summary_out.csv')