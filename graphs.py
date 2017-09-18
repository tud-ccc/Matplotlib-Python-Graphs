from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from matplotlib.legend_handler import (HandlerLineCollection,
                                       HandlerTuple)
from matplotlib import rc, font_manager

import matplotlib.collections as mcol
from matplotlib.lines import Line2D

import csv
import sys
import numpy as np
import glob
import copy

#import pandas as pd
from datetime import datetime
from decimal import *

import statistics as stat
from scipy import stats

#import seaborn as sns

import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.lines as mpllines
import matplotlib.backends.backend_pdf

##########################################
#Graphs variables declaration
##########################################
#color =     ['#ffffff', '#cfcecf', '#918c8c', '#6f6d6d']
#edgecolor = ['#cfcecf', '#918c8c', '#6f6d6d', '#504d4d']

#color =     ['#ffffff', '#81BEF7', '#D8D8D8', '#424242', '#FFFFFF', '#6f6d6d']
#edgecolor = ['#cfcecf', '#045FB4', '#120A2A', '#120A2A', '#120A2A', '#504d4d']
#hatch =     [ "/////", " ", " ", "oo", ".....", " "]
marker = ['D', 'v','H', 'v', '*', '>', '<', 'o', 'x']
color = []
edgecolor = []
hatch =     []



Y_min =     [0,0,0,0,0,0,0,0,0]
Y_max =     [9500,9500,100,100,9500,9500,100,100,9500]

Y_min =     [0,0,0,0,0,0,0,0,0]
Y_max_stacked =     [3500*6,3500*6,100*6,100*6,3500*6,3500*6,100*6,100*6,3500*6]

##########################################
#Variables declaration
##########################################

file_names = []
dirs_names = []

graph_last_index_values = []
yaxis_array = []
xaxis_array = []

legends_names_array= []
parameters_names_array = []
null_array = []

rb_hits_data = []
rb_miss_data = []
rb_hitrate_data = []
rb_missrate_data = []

drc_hits_data = []
drc_miss_data = []
drc_hitrate_data = []
drc_missrate_data = []

tagCacheHits_data = []
tagCacheMisses_data = []
add_1_data = []

rb_hits_last_index_values = []
rb_miss_last_index_values = []
rb_hitrate_last_index_values = []
rb_missrate_last_index_values = []
drc_hits_last_index_values = []
drc_miss_last_index_values = []
drc_hitrate_last_index_values = []
drc_missrate_last_index_values = []

tagCacheHits_last_index_values = []
tagCacheMisses_last_index_values = []

add_1_last_index_values = []
parameters_checks = []
parameters_range= []
parameters_file_norm= []
x_axis= []

parameters_default = []
parameters_max = []
parameters_min = []
parameters_avg_hm = []
parameters_ref_legend_file = []
parameters_scaled_array = []
parameters_graph = []
parameters_y_axis_mark= []
parameters_x_axis_rot= []

array_temp1 = []
array_temp2 = []
array_temp3 = []
array_temp4 = []
array_temp5 = []
array_temp6 = []
array_temp7 = []
array_temp8 = []
array_temp9 = []
array_temp10 = []
array_temp11 = []



#------------------------------------------------------------------------------------------------------------------#
##########################################
# Data Array Operation
##########################################
def array_manupulation(rb_hits_data,rb_miss_data,rb_hitrate_data,rb_missrate_data,drc_hits_data,drc_miss_data,drc_hitrate_data,drc_missrate_data,tagCacheHits_data,tagCacheMisses_data,add_1_data):

	
	#print (rb_hits_data)
	a = len(rb_hits_data)
	b = len(rb_miss_data)
	c = len(rb_hitrate_data)
	d = len(rb_missrate_data)
	e = len(drc_hits_data)
	f = len(drc_miss_data)
	g = len(drc_hitrate_data)
	h = len(drc_missrate_data)
	i = len(tagCacheHits_data)
	j = len(tagCacheMisses_data)
	k = len(add_1_data)
	
	
	"""
	for r in range(c): 
		rb_missrate_data[r] = 100 - rb_hitrate_data[r]
	for p in range(g-1): 
		drc_missrate_data[p] = 100 - drc_hitrate_data[p]
	print rb_hitrate_data
	print rb_missrate_data
	#print drc_missrate_data
	"""
	list = [a, b, c, d, e, f, g, h, i, j, k]
	#print list
	
	if a > 0:
		data = rb_hits_data[a-1]
		rb_hits_last_index_values.append(data)
	else:
		rb_hits_last_index_values.append('0')
	if b > 0:
		data = rb_miss_data[b-1]
		rb_miss_last_index_values.append(data)
	else:
		rb_miss_last_index_values.append('0')
	if c > 0:
		data = rb_hitrate_data[c-1]
		rb_hitrate_last_index_values.append(data)
	else:
		rb_hitrate_last_index_values.append('0')
	if d > 0:
		data = rb_missrate_data[d-1]
		rb_missrate_last_index_values.append(data)
	else:
		rb_missrate_last_index_values.append('0')
	if e > 0:
		data = drc_hits_data[e-1]
		drc_hits_last_index_values.append(data)
	else:
		drc_hits_last_index_values.append('0')
	if f > 0:
		data = drc_miss_data[f-1]
		drc_miss_last_index_values.append(data)
	else:
		drc_miss_last_index_values.append('0')
	if g > 0:
		data = drc_hitrate_data[g-1]
		drc_hitrate_last_index_values.append(data)
	else:
		drc_hitrate_last_index_values.append('0')
	if h > 0:
		data = drc_missrate_data[h-1]
		drc_missrate_last_index_values.append(data)
	else:
		drc_missrate_last_index_values.append('0')
	if i > 0:
		data = tagCacheHits_data[i-1]
		tagCacheHits_last_index_values.append(data)
	else:
		tagCacheHits_last_index_values.append('0')
	if j > 0:
		data = tagCacheMisses_data[j-1]
		tagCacheMisses_last_index_values.append(data)
	else:
		tagCacheMisses_last_index_values.append('0')
	if k > 0:
		data = add_1_data[j-1]
		add_1_last_index_values.append(data)
	else:
		add_1_last_index_values.append('0')


	#print tagCacheMisses_data
	#print tagCacheMisses_last_index_values

	temp = max(list)

	x = '-'
	for n in range(temp): 
		null_array.append(x)
		#null_array.append([])

	rb_hits_data.extend(null_array)
	rb_miss_data.extend(null_array)
	rb_hitrate_data.extend(null_array)
	rb_missrate_data.extend(null_array)
	drc_hits_data.extend(null_array)
	drc_miss_data.extend(null_array)
	drc_hitrate_data.extend(null_array)
	drc_missrate_data.extend(null_array)
	tagCacheHits_data.extend(null_array)
	tagCacheMisses_data.extend(null_array)
	add_1_data.extend(null_array)
	return temp

##########################################


#######################################
#Reading parameters file 
#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
def read_parameters_file():

	parameters_file = open("parameters.txt", "r")


	for line in parameters_file:
		if not line.startswith("-") and not line.startswith("'") and not line.startswith("!") and not line.startswith("="):
			str = line

			data = str.split( )[0]
			parameters_names_array.append(data)

			data = str.split( )[1]
			parameters_graph.append(int(data))

			data = str.split( )[2]
			parameters_scaled_array.append(data)

			data = str.split( )[3]
			parameters_ref_legend_file.append(int(data))

			data = str.split( )[4]
			parameters_avg_hm.append(data)

			data = str.split( )[5]
			parameters_min.append(int(data))

			data = str.split( )[6]
			parameters_max.append(int(data))

			data = str.split( )[7]
			parameters_default.append(data)

			data = str.split( )[8]
			parameters_y_axis_mark.append(data)

			data = str.split( )[9]
			parameters_x_axis_rot.append(int(data))

		elif line.startswith("-"):
			str = line
			#print str.split( )[1]

			data = str.split( )[1]
			parameters_checks.append(data)

		elif line.startswith("'"):
			str = line
			#print str.split( )[1]

			data = str.split( )[1]
			parameters_range.append(data)

		elif line.startswith("!"):
			str = line
			#print str.split( )[1]

			data = str.split( )[1]
			parameters_file_norm.append(data)


#print parameters_checks
##########################################
#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
def read_legends_file():

	legends_file = open("legends.txt", "r")

	#Reading legends file 

	for line in legends_file:
		str = line
		data = str.split('\n')[0]
		legends_names_array.append(data)
 
########################################
#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
def read_format_file():

	format_file = open("format.txt", "r")

	#Reading format file 

	for line in format_file:
		if not line.startswith("'"):
			str = line

			data = str.split('\n')[0]
			data = data.split(',')[0]
			color.append(data)

			data = str.split(',')[1]
			if data != '\n':
				edgecolor.append(data)

			data = str.split(',')[2]
			if data != '\n':
				hatch.append(data)

	for k in range(len(hatch)):
		hatch[k] = hatch[k].rstrip('\n')
########################################
#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
#Reading axis file 
def read_axis_file():

	axis_file = open("axis.txt", "r")

	for line in axis_file:
		if not line.startswith("'"):
			str = line
			#print str.split( )[1]

			data = str.split('\n')[0]
			data = data.split(',')[0]
			yaxis_array.append(data)

			data = str.split(',')[1]
			if data != '\n':
				x_axis.append(data)

	for k in range(len(x_axis)):
		x_axis[k] = x_axis[k].rstrip('\n')


#######################################
def data_extraction_logfiles():
	#print (parameters_names_array)
	#Extracting different parameters values from log file
	path = "Script/"
	global dirs_names
	dirs_names = sorted(glob.glob(path + '*')) #key=numericalSort
	#######################################
	#------------------------------------------------------------------------------------------------------------------#
	#
	#
	#
	#------------------------------------------------------------------------------------------------------------------#

	#---------------------------------------#
	for k in range(len(dirs_names)):
	#---------------------------------------#
		global file_names
		file_names = sorted(glob.glob(dirs_names[k] + "/*.log"))
		#print(file_names)
		#---------------------------------------#
		for j in range(len(file_names)):
		#---------------------------------------#
			rb_hits_data = []
			rb_miss_data = []
			rb_hitrate_data = []
			rb_missrate_data = []

			drc_hits_data = []
			drc_miss_data = []
			drc_hitrate_data = []
			drc_missrate_data = []

			tagCacheHits_data = []
			tagCacheMisses_data = []
			add_1_data = []
		#---------------------------------------#
			searchfile = open(file_names[j], "r")
			for line in searchfile:
			    if len(parameters_names_array) >= 1 and parameters_names_array[0] in line: 
			       str = line
			       #print (rb_hits_data)
			       data = str.split( )[1]
			       rb_hits_data.append(int(data, 16))
		#---------------------------------------#
			    if len(parameters_names_array) >= 2 and parameters_names_array[1] in line: 
			       str = line

			       data = str.split( )[1]
			       rb_miss_data.append(int(data, 16))
		#---------------------------------------#
			    if len(parameters_names_array) >= 3 and parameters_names_array[2] in line: 
			       str = line

			       data = str.split( )[1]
			       data = round(Decimal(data)*100)
			       #print data
			       rb_hitrate_data.append(data)
		#---------------------------------------#
			    if len(parameters_names_array) >= 4 and parameters_names_array[3] in line: 
			       str = line

			       data = str.split( )[1]
			       rb_missrate_data.append(int(data, 16))
		#---------------------------------------#
			    if len(parameters_names_array) >= 5 and parameters_names_array[4] in line: 
			       str = line

			       data = str.split( )[1]
			       drc_hits_data.append(int(data, 16))
		#---------------------------------------#
			    if len(parameters_names_array) >= 6 and parameters_names_array[5] in line: 
			       str = line

			       data = str.split( )[1]
			       drc_miss_data.append(int(data, 16))
		#---------------------------------------#
			    if len(parameters_names_array) >= 7 and parameters_names_array[6] in line: 
			       str = line

			       data = str.split( )[1]
			       data = round(Decimal(data)*100)
			       #print data
			       drc_hitrate_data.append(data)
		#---------------------------------------#
			    if len(parameters_names_array) >= 8 and parameters_names_array[7] in line: 
			       str = line

			       data = str.split( )[1]
			       drc_missrate_data.append(int(data, 16))
		#---------------------------------------#
			    if len(parameters_names_array) >= 9 and parameters_names_array[8] in line: 
			       str = line

			       data = str.split( )[1]
			       tagCacheHits_data.append(int(data, 16))
		#---------------------------------------#
			    if len(parameters_names_array) >= 10 and parameters_names_array[9] in line: 
			       str = line
			    
			       data = str.split( )[1]
			       tagCacheMisses_data.append(int(data, 16))
		#---------------------------------------#
			    if len(parameters_names_array) >= 11 and parameters_names_array[10] in line: 
			       str = line
			    
			       data = str.split( )[1]
			       add_1_data.append(int(data, 16))
		#---------------------------------------#
		# Add more if cases if parameters file contain more parameters
		#---------------------------------------#
			if not rb_missrate_data:
			   data = 100 - rb_hitrate_data[0]
			   rb_missrate_data.append(data)
			if not drc_missrate_data:
			   data = 100 - drc_hitrate_data[0]
			   drc_missrate_data.append(data)
			temp = array_manupulation(rb_hits_data, rb_miss_data, rb_hitrate_data, rb_missrate_data, drc_hits_data, drc_miss_data, drc_hitrate_data, drc_missrate_data, tagCacheHits_data, tagCacheMisses_data, add_1_data)
			#print (rb_hits_last_index_values)

#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#	This module reads the data from CSV file to plot difeerent graphs.
#	reading_csv()
#	Tanveer Ahmad
#------------------------------------------------------------------------------------------------------------------#
def reading_csv():
	temp_buffer = []
	temp_buffer2D = []
	splitted_list2D= []
	rb_hits_last_index_values = []
	rb_miss_last_index_values = []
	rb_hitrate_last_index_values = []
	rb_missrate_last_index_values = []
	drc_hits_last_index_values = []
	drc_miss_last_index_values = []
	drc_hitrate_last_index_values = []
	drc_missrate_last_index_values = []

	tagCacheHits_last_index_values = []
	tagCacheMisses_last_index_values = []
	dir_no = 0
	row_check = True
	with open("Graphs/overall_csv.csv",'r') as csvFile:
		reader=csv.reader(csvFile,delimiter=',')
		next(reader)
		for row in reader:
		    if row_check == True: 
		        temp_row = row
		        row_buffer_offset = int (((len(row)/2) - 2))
		        global file_names
		        file_names = [elem for elem in range(row_buffer_offset)]
		        row_check = False
		    if len(row) > 1:
		        temp_buffer.extend(row[int (((len(row)/2) + 2)):])
		        dir_no += 1
		    else:
		        temp_buffer = [float(i) for i in temp_buffer[row_buffer_offset:]]
		        #print(temp_buffer)
		        temp_buffer2D.append(temp_buffer)
		        global dirs_names
		        dirs_names = [elem for elem in range(dir_no-2)]
		        dir_no = 0
		        temp_buffer= []

		temp_buffer = [float(i) for i in temp_buffer[row_buffer_offset:]]
		temp_buffer2D.append(temp_buffer)

		for j in range(sum(1 for x in parameters_graph if x == 1)):    
		    
		    splitted_list = split_list_simple(temp_buffer2D[j], len(dirs_names)+1)
		    splitted_list2D.append(splitted_list)
	return splitted_list2D

#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
def percentage(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x) + '%'
#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
def simple(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x)

#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
def split_list(alist, wanted_parts, scaled, ref_file_index, csv_writing, writer):

    extended = []
    extended2d = []
    transposed = []
    length = len(alist)

    splitted_list = [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]

    if scaled == True:
	    for e in range(len(dirs_names)):
	        extended = splitted_list[e]
	        new = [round(float(1 + ((ele - extended[ref_file_index-1]) / extended[ref_file_index-1])), 5) for ele in extended]
	        extended2d.append(new)
    #print (extended2d)
    if csv_writing == True:
	    for file in range(len(dirs_names)):
		    rwotowrite = [' '] + [x_axis[file]] + [elem for elem in splitted_list[file]] + [' '] + [' '] + [elem_1 for elem_1 in extended2d[file]]
		    writer.writerow(rwotowrite)

    for i in range(len(file_names)):
	    if scaled == True:
		    transposed.append([row[i] for row in extended2d])
	    else:
		    transposed.append([row[i] for row in splitted_list])
    #print (transposed)
    return transposed

		
#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
def split_list_simple(alist, wanted_parts):

    transposed = []
    length = len(alist)

    splitted_list = [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]

    for i in range(len(file_names)):
        transposed.append([row[i] for row in splitted_list])
    
    return transposed
	
#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs
#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
my_objects = []
class MyClass(object):
    def __init__(self, obj):
        self.x = obj

#------------------------------------------------------------------------------------------------------------------#
def graphs_generation():
	if sys.argv[2] == 'direct':
		all_data_values_2D = reading_csv()
#------------------------------------------------------------------------------------------------------------------#
	if sys.argv[1] == 'bar':
	#------------------------------------------------------------------------------------------------------------------#
		with matplotlib.backends.backend_pdf.PdfPages('Graphs/graphs_dram_caches_bars.pdf') as pdf:
		#------------------------------------------------------------------------------------------------------------------#	
			if not sys.argv[2] == 'direct':
				csv_writing = True
				#
				f = open("Graphs/overall_csv.csv", 'wt')
				writer = csv.writer(f)
		#------------------------------------------------------------------------------------------------------------------#
			for x in range(len(parameters_graph)):
				if x == 0:
					graph_last_index_values = copy.deepcopy(rb_hits_last_index_values)
				elif x == 1:
					graph_last_index_values = copy.deepcopy(rb_miss_last_index_values)
				elif x == 2:
					graph_last_index_values = copy.deepcopy(rb_hitrate_last_index_values)
				elif x == 3:
					graph_last_index_values = copy.deepcopy(rb_missrate_last_index_values)
				elif x == 4:
					graph_last_index_values = copy.deepcopy(drc_hits_last_index_values)
				elif x == 5:
					graph_last_index_values = copy.deepcopy(drc_miss_last_index_values)
				elif x == 6:
					graph_last_index_values = copy.deepcopy(drc_hitrate_last_index_values)
				elif x == 7:
					graph_last_index_values = copy.deepcopy(drc_missrate_last_index_values)
				elif x == 8:
					graph_last_index_values = copy.deepcopy(tagCacheHits_last_index_values)
				elif x == 9:
					graph_last_index_values = copy.deepcopy(tagCacheMisses_last_index_values)
				elif x == 10:
					graph_last_index_values = copy.deepcopy(add_1_last_index_values)

				if parameters_graph[x] == 1:
					if not sys.argv[2] == 'direct':
						avg_hm = []
						header = [' '] + [parameters_names_array[x]] + [leg for leg in legends_names_array] + [' '] + [' '] + [leg for leg in legends_names_array]
						#print (header)
						writer.writerow([' '])
						writer.writerow( header )

					ymax = parameters_max[x]
					ymin = parameters_min[x]

					if parameters_avg_hm[x] == 'Y':
						N = len(dirs_names) + 1 
						x_axis[N-1] = 'Average' 
					elif parameters_avg_hm[x] == 'N':
						N = len(dirs_names) + 1 
						x_axis[N-1] = 'H.Mean' 
					else:
						N = len(dirs_names) #3
					ind = np.arange(N)  # the x locations for the groups
					width = 0.18#len(dirs_names) / (100.0 / (len(file_names)+1))  #20.0 #0.15       # the width of the bars
					fig, ax = plt.subplots(figsize=(5.5, 1.8), dpi=600) #(num=None, figsize=(16, 12), dpi=180, facecolor='w', edgecolor='k')
					fig.canvas.set_window_title(yaxis_array[x])

					pos1 = ax.get_position() # get the original position 
					pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height] 
					ax.set_position(pos2) # set a new position
					
					if not sys.argv[2] == 'direct':
						if parameters_scaled_array[x] == 'Y':
							splitted_list = split_list(graph_last_index_values, len(dirs_names), True, parameters_ref_legend_file[x], csv_writing, writer)
						else:
							splitted_list = split_list(graph_last_index_values, len(dirs_names), False, parameters_ref_legend_file[x], csv_writing, writer)
						#Harmonic Mean
						if parameters_avg_hm[x] == 'N':
						    for e in range(len(file_names)):
						        new = stats.hmean(splitted_list[e])
						        splitted_list[e].append(new)
						        avg_hm.append(new)
						        heading_avg_hm = 'H.Mean'

						#Average Graph
						elif parameters_avg_hm[x] == 'Y':
						    for e in range(len(file_names)):
						        new = float(sum(splitted_list[e])/len(splitted_list[e]))
						        splitted_list[e].append(new)
						        avg_hm.append(new)
						        heading_avg_hm = 'Average'

					#####################################
					#print (splitted_list)
					if sys.argv[2] == 'direct':
					        splitted_list = all_data_values_2D[x]
					#print (splitted_list)
					#####################################
					if not sys.argv[2] == 'direct':
						avg_hm_line = [' '] + [' '] + [' ' for leg in legends_names_array] + [' '] + [heading_avg_hm] + [leg for leg in avg_hm]
						writer.writerow( avg_hm_line )

					#print (splitted_list)
					rbhitsFs = []
					for y in range(len(file_names)):
					#Benchmarks B1  B2  B3 ...
						rbhitsF = [bar for bar in splitted_list[y]]
						rbhitsFs.append(rbhitsF)

						#Scalled Graph
						if parameters_default[x] == 'Y':
							max(bar for bar in splitted_list[y])#ymax = 

					for z in range(len(file_names)):
						ax.bar(ind + (width*(z*0.65)), rbhitsFs[z], width*0.65, color=color[z], edgecolor=edgecolor[z], yerr=None, hatch=hatch[z])

					if not parameters_y_axis_mark[x] == 'None':
						formatter = FuncFormatter(percentage)
						ax.yaxis.set_major_formatter(formatter)
					else:
						formatter = FuncFormatter(simple)
						ax.yaxis.set_major_formatter(formatter)

					# add some text for labels, title and axes ticks
					ax.set_ylabel(yaxis_array[x], fontsize=8, color='black', family='serif')
					#ax.set_xlabel('Benchmarks', fontsize=10, color='black')
					#ax.set_title('DRAM Cache Configurations Logs\n')
					ax.set_ylim([ymin,ymax])
					xticks = ind + (0.0585 * (len(file_names)))
					ax.set_xticks(xticks)
					ax.set_xticklabels(([axis for axis in x_axis]), fontsize=6, family='serif', rotation = parameters_x_axis_rot[x])#
					#ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
					ax.ticklabel_format(family='serif')

					# Shrink current axis by 20%
					box = ax.get_position()
					ax.set_position([box.x0, box.y0, box.width * 1.2, box.height])
					# Hide the right and top spines
					ax.spines['right'].set_visible(True)
					ax.spines['top'].set_visible(True)
					ax.spines['bottom'].set_linewidth(0.5)
					ax.spines['left'].set_linewidth(0.5)
					ax.spines['right'].set_linewidth(0.5)
					ax.spines['top'].set_linewidth(0.5)

					ax.spines['bottom'].set_color('#848484')
					ax.spines['left'].set_color('#848484') 
					ax.spines['right'].set_color('#848484')
					ax.spines['top'].set_color('#848484') 
			
					# Only show ticks on the left and bottom spines
					ax.yaxis.set_ticks_position('left')
					ax.xaxis.set_ticks_position('bottom')
					# Try to set the tick markers to extend outward from the axes, R-style
					for line in ax.get_xticklines():
					    line.set_marker(mpllines.TICKDOWN)
					    line.set_markeredgewidth(0.5)
					    line.set_markeredgecolor('#848484')

					for line in ax.get_yticklines():
					    line.set_marker(mpllines.TICKLEFT)
					    line.set_markeredgewidth(0.5)
					    line.set_markeredgecolor('#848484') 
			
					ticks_font = font_manager.FontProperties(family='serif', style='normal',size=6, weight='normal', stretch='normal')
					for label in ax.get_yticklabels():
		    			    label.set_fontproperties(ticks_font)
		
					L = ax.legend([leg for leg in legends_names_array], loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True,  fontsize=6) #ncol=len(file_names)/2,shadow=True,
					L.get_frame().set_edgecolor('#848484')
					L.get_frame().set_linewidth(0.5)
					plt.setp(L.texts, family='serif')
					ax.set_axisbelow(True)

					ax.yaxis.grid(color='gray', linestyle='-', linewidth=.5)
					plt.tight_layout()
					#ax.legend( [leg for leg in legends_names_array], bbox_to_anchor=(1.04,1), loc="best")	    
					#ax.axvspan(9.5, 9.01, facecolor='g', alpha=0.5)
					#ax.axhspan(9.5, 9.01, facecolor='g', alpha=0.5)
			
					plt.axvline(x=N - 1.2, color='gray', linestyle='--', linewidth=.5)
			
					#ax2 = ax.twinx()
					#ax2.plot([4491, 10579, 6400], 'x-', markeredgewidth=2, color='#7e7e7e', linewidth=2, markersize=8)

					#print ([max_sl for max_sl in splitted_list])
					pdf.savefig()  # saves the current figure into a pdf page
					plt.savefig('Graphs/' + yaxis_array[x] + '_bar.pdf')
			


					#plt.show()
					#mpl.style.use('seaborn')

	#------------------------------------------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------------------------------------------#



	#------------------------------------------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------------------------------------------#
	elif sys.argv[1] == 'stack':
	#------------------------------------------------------------------------------------------------------------------#
		with matplotlib.backends.backend_pdf.PdfPages('Graphs/graphs_dram_caches_stacked.pdf') as pdf:	#------------------------------------------------------------------------------------------------------------------												        
			if not sys.argv[2] == 'direct':
				csv_writing = True
				#
				f = open("Graphs/overall_csv.csv", 'wt')
				writer = csv.writer(f)
		#------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------#
			for x in range(len(parameters_graph)):
				if x == 0:
					graph_last_index_values = copy.deepcopy(rb_hits_last_index_values)
					#print (rb_hits_last_index_values)
				elif x == 1:
					graph_last_index_values = copy.deepcopy(rb_miss_last_index_values)
				elif x == 2:
					graph_last_index_values = copy.deepcopy(rb_hitrate_last_index_values)
				elif x == 3:
					graph_last_index_values = copy.deepcopy(rb_missrate_last_index_values)
				elif x == 4:
					graph_last_index_values = copy.deepcopy(drc_hits_last_index_values)
				elif x == 5:
					graph_last_index_values = copy.deepcopy(drc_miss_last_index_values)
				elif x == 6:
					graph_last_index_values = copy.deepcopy(drc_hitrate_last_index_values)
				elif x == 7:
					graph_last_index_values = copy.deepcopy(drc_missrate_last_index_values)
				elif x == 8:
					graph_last_index_values = copy.deepcopy(tagCacheHits_last_index_values)
				elif x == 9:
					graph_last_index_values = copy.deepcopy(tagCacheMisses_last_index_values)
				elif x == 10:
					graph_last_index_values = copy.deepcopy(add_1_last_index_values)
		
				if parameters_graph[x] == 1:
			
					ymax = parameters_max[x] #* len(file_names)
					ymin = parameters_min[x] #* len(file_names)
					if not sys.argv[2] == 'direct':
						avg_hm = [] 
						header = [' '] + [parameters_names_array[x]] + [leg for leg in legends_names_array] + [' '] + [' '] + [leg for leg in legends_names_array]
						#print (header)
						writer.writerow([' '])
						writer.writerow( header )

					if parameters_avg_hm[x] == 'Y':
						N = len(dirs_names) + 1 
						x_axis[N-1] = 'Average' 
					elif parameters_avg_hm[x] == 'N':
						N = len(dirs_names) + 1 
						x_axis[N-1] = 'H.Mean' 
					else:
						N = len(dirs_names) #3

					ind = np.arange(N)  # the x locations for the groups
					width = 0.55#len(dirs_names) / (100.0 / (len(file_names)+1))  #20.0 #0.15       # the width of the bars
					fig, ax = plt.subplots(figsize=(3.2, 1.5), dpi=600) #(num=None, figsize=(16, 12), dpi=180, facecolor='w', edgecolor='k')
					fig.canvas.set_window_title(yaxis_array[x])

					pos1 = ax.get_position() # get the original position 
					pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height] 
					ax.set_position(pos2) # set a new position
					if not sys.argv[2] == 'direct':
						if parameters_scaled_array[x] == 'Y':
							splitted_list = split_list(graph_last_index_values, len(dirs_names), True, parameters_ref_legend_file[x], csv_writing, writer)
						else:
							splitted_list = split_list(graph_last_index_values, len(dirs_names), False, parameters_ref_legend_file[x], csv_writing, writer)
						#print splitted_list

						#Harmonic Mean
						if parameters_avg_hm[x] == 'N':
						    for e in range(len(file_names)):
						        new = stats.hmean(splitted_list[e])
						        splitted_list[e].append(new)
						        avg_hm.append(new)
						        heading_avg_hm = 'H.Mean'
	
						#Average Graph
						elif parameters_avg_hm[x] == 'Y':
						    for e in range(len(file_names)):
						        new = float(sum(splitted_list[e])/len(splitted_list[e]))
						        splitted_list[e].append(new)
						        avg_hm.append(new)
						        heading_avg_hm = 'Average'

						avg_hm_line = [' '] + [' '] + [' ' for leg in legends_names_array] + [' '] + [heading_avg_hm] + [leg for leg in avg_hm]
						writer.writerow( avg_hm_line )

					#####################################
					#print (splitted_list)
					if sys.argv[2] == 'direct':
					        splitted_list = all_data_values_2D[x]
					#print (splitted_list)
					#####################################

					rbhitsFs = []
					for y in range(len(file_names)):
					#Benchmarks B1  B2  B3
						rbhitsF = [bar for bar in splitted_list[y]]
						rbhitsFs.append(rbhitsF)

					for z in range(len(file_names)):
						if z == 0:
							bottom_val = 0
						elif z == 1:
							bottom_val = rbhitsFs[0]
						elif z == 2:
							bottom_val = [i+j for i,j in zip(rbhitsFs[0],rbhitsFs[1])]
						elif z == 3:
							bottom_val = [i+j+k for i,j,k in zip(rbhitsFs[0],rbhitsFs[1],rbhitsFs[2])]
						elif z == 4:
							bottom_val = [i+j+k+l for i,j,k,l in zip(rbhitsFs[0],rbhitsFs[1],rbhitsFs[2],rbhitsFs[3])]
						elif z == 5:
							bottom_val = [i+j+k+l+m for i,j,k,l,m in zip(rbhitsFs[0],rbhitsFs[1],rbhitsFs[2],rbhitsFs[3],rbhitsFs[4])]
						elif z == 6:
							bottom_val = [i+j+k+l+m+n for i,j,k,l,m,n in zip(rbhitsFs[0],rbhitsFs[1],rbhitsFs[2],rbhitsFs[3],rbhitsFs[4],rbhitsFs[5])]

						ax.bar(ind, rbhitsFs[z], width, bottom=bottom_val, color=color[z], edgecolor=edgecolor[z], yerr=None, hatch=hatch[z])
		

					if not parameters_y_axis_mark[x] == 'None':
						formatter = FuncFormatter(percentage)
						ax.yaxis.set_major_formatter(formatter)
					else:
						formatter = FuncFormatter(simple)
						ax.yaxis.set_major_formatter(formatter)

					# add some text for labels, title and axes ticks
					ax.set_ylabel(yaxis_array[x], fontsize=8, color='black', family='serif')
					#ax.set_xlabel('Benchmarks', fontsize=10, color='black')
					#ax.set_title('DRAM Cache Configurations Logs\n')
					ax.set_ylim([ymin,ymax])
					xticks = ind + (0.0485 * (len(file_names)))
					ax.set_xticks(xticks)
					ax.set_xticklabels(([axis for axis in x_axis]), fontsize=6, family='serif', rotation = parameters_x_axis_rot[x])#
					#ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
					ax.ticklabel_format(family='serif')

					# Shrink current axis by 20%
					box = ax.get_position()
					ax.set_position([box.x0, box.y0, box.width * 1.3, box.height])
					# Hide the right and top spines
					ax.spines['right'].set_visible(True)
					ax.spines['top'].set_visible(True)
					ax.spines['bottom'].set_linewidth(0.5)
					ax.spines['left'].set_linewidth(0.5)
					ax.spines['right'].set_linewidth(0.5)
					ax.spines['top'].set_linewidth(0.5)

					ax.spines['bottom'].set_color('#848484')
					ax.spines['left'].set_color('#848484') 
					ax.spines['right'].set_color('#848484')
					ax.spines['top'].set_color('#848484') 
			
					# Only show ticks on the left and bottom spines
					ax.yaxis.set_ticks_position('left')
					ax.xaxis.set_ticks_position('bottom')
					# Try to set the tick markers to extend outward from the axes, R-style
					for line in ax.get_xticklines():
					    line.set_marker(mpllines.TICKDOWN)
					    line.set_markeredgewidth(0.5)
					    line.set_markeredgecolor('#848484')

					for line in ax.get_yticklines():
					    line.set_marker(mpllines.TICKLEFT)
					    line.set_markeredgewidth(0.5)
					    line.set_markeredgecolor('#848484') 
			
					ticks_font = font_manager.FontProperties(family='serif', style='normal',size=6, weight='normal', stretch='normal')
					for label in ax.get_yticklabels():
		    			    label.set_fontproperties(ticks_font)
		
					L = ax.legend([leg for leg in legends_names_array], loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True,  fontsize=6) #ncol=len(file_names)/2,shadow=True,
					L.get_frame().set_edgecolor('#848484')
					L.get_frame().set_linewidth(0.5)
					plt.setp(L.texts, family='serif')
					ax.set_axisbelow(True)

					ax.yaxis.grid(color='gray', linestyle='-', linewidth=.5)
					plt.tight_layout()
					#ax.legend( [leg for leg in legends_names_array], bbox_to_anchor=(1.04,1), loc="best")	    
					#ax.axvspan(9.5, 9.01, facecolor='g', alpha=0.5)
					#ax.axhspan(9.5, 9.01, facecolor='g', alpha=0.5)
			
					plt.axvline(x=N - 1.2, color='gray', linestyle='--', linewidth=.5)
			
					#ax2 = ax.twinx()
					#ax2.plot([4491, 10579, 6400], 'x-', markeredgewidth=2, color='#7e7e7e', linewidth=2, markersize=8)

					#print ([max_sl for max_sl in splitted_list])
					pdf.savefig()  # saves the current figure into a pdf page
					plt.savefig('Graphs/' + yaxis_array[x] + '_stack.pdf')
					#plt.show()
	#------------------------------------------------------------------------------------------------------------------#/net/home/tahmad/Documents/Python/Script/
	#------------------------------------------------------------------------------------------------------------------#

	#------------------------------------------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------------------------------------------#
	#------------------------------------------------------------------------------------------------------------------#
	elif sys.argv[1] == 'line':
	#------------------------------------------------------------------------------------------------------------------#
		with matplotlib.backends.backend_pdf.PdfPages('Graphs/graphs_dram_caches_lines.pdf') as pdf:		  		#---------------------------------------------------------------------------------------------------												        
			if not sys.argv[2] == 'direct':
				csv_writing = True
				#
				f = open("Graphs/overall_csv.csv", 'wt')
				writer = csv.writer(f)
		#------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------#
			for x in range(len(parameters_graph)):
				if x == 0:
					graph_last_index_values = copy.deepcopy(rb_hits_last_index_values)
					#print (rb_hits_last_index_values)
				elif x == 1:
					graph_last_index_values = copy.deepcopy(rb_miss_last_index_values)
				elif x == 2:
					graph_last_index_values = copy.deepcopy(rb_hitrate_last_index_values)
				elif x == 3:
					graph_last_index_values = copy.deepcopy(rb_missrate_last_index_values)
				elif x == 4:
					graph_last_index_values = copy.deepcopy(drc_hits_last_index_values)
				elif x == 5:
					graph_last_index_values = copy.deepcopy(drc_miss_last_index_values)
				elif x == 6:
					graph_last_index_values = copy.deepcopy(drc_hitrate_last_index_values)
				elif x == 7:
					graph_last_index_values = copy.deepcopy(drc_missrate_last_index_values)
				elif x == 8:
					graph_last_index_values = copy.deepcopy(tagCacheHits_last_index_values)
				elif x == 9:
					graph_last_index_values = copy.deepcopy(tagCacheMisses_last_index_values)
				elif x == 10:
					graph_last_index_values = copy.deepcopy(add_1_last_index_values)
		
				if parameters_graph[x] == 1:
			
					ymax = parameters_max[x] 
					ymin = parameters_min[x] 
					if not sys.argv[2] == 'direct':
						avg_hm = [] 
						header = [' '] + [parameters_names_array[x]] + [leg for leg in legends_names_array] + [' '] + [' '] + [leg for leg in legends_names_array]
						#print (header)
						writer.writerow([' '])
						writer.writerow( header )

					if parameters_avg_hm[x] == 'Y':
						N = len(dirs_names) + 1 
						x_axis[N-1] = 'Average' 
					elif parameters_avg_hm[x] == 'N':
						N = len(dirs_names) + 1 
						x_axis[N-1] = 'H.Mean' 
					else:
						N = len(dirs_names) #3

					ind = np.arange(N)  # the x locations for the groups
					width = 0.55#len(dirs_names) / (100.0 / (len(file_names)+1))  #20.0 #0.15       # the width of the bars
					fig, ax = plt.subplots(figsize=(3.2, 1.5), dpi=600) #(num=None, figsize=(16, 12), dpi=180, facecolor='w', edgecolor='k')
					fig.canvas.set_window_title(yaxis_array[x])

					pos1 = ax.get_position() # get the original position 
					pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height] 
					ax.set_position(pos2) # set a new position
					if not sys.argv[2] == 'direct':
						if parameters_scaled_array[x] == 'Y':
							splitted_list = split_list(graph_last_index_values, len(dirs_names), True, parameters_ref_legend_file[x], csv_writing, writer)
						else:
							splitted_list = split_list(graph_last_index_values, len(dirs_names), False, parameters_ref_legend_file[x], csv_writing, writer)
						#print splitted_list

						#Harmonic Mean
						if parameters_avg_hm[x] == 'N':
						    for e in range(len(file_names)):
						        new = stats.hmean(splitted_list[e])
						        splitted_list[e].append(new)
						        avg_hm.append(new)
						        heading_avg_hm = 'H.Mean'

						#Average Graph
						elif parameters_avg_hm[x] == 'Y':
							#average_list = [int(sum(l))/len(l) for l in zip(*splitted_list)]
							#splitted_list.append(average_list)
						    for e in range(len(file_names)):
						        new = float(sum(splitted_list[e])/len(splitted_list[e]))
						        splitted_list[e].append(new)
						        avg_hm.append(new)
						        heading_avg_hm = 'Average'
						
						avg_hm_line = [' '] + [' '] + [' ' for leg in legends_names_array] + [' '] + [heading_avg_hm] + [leg for leg in avg_hm]
						writer.writerow( avg_hm_line )

					# evenly sampled time at 200ms intervals
					t = np.arange(0,0.5)

					#####################################
					#print (splitted_list)
					if sys.argv[2] == 'direct':
					        splitted_list = all_data_values_2D[x]
					#print (splitted_list)
					#####################################
			
					for i in range(len(file_names)):
					# red dashes, blue squares and green triangles
						plt.plot(splitted_list[i], marker = marker[i], linewidth=1, linestyle="-", color=color[i], markeredgecolor=edgecolor[i])
						#print (splitted_list[i])

					if not parameters_y_axis_mark[x] == 'None':
						formatter = FuncFormatter(percentage)
						ax.yaxis.set_major_formatter(formatter)
					else:
						formatter = FuncFormatter(simple)
						ax.yaxis.set_major_formatter(formatter)

					# add some text for labels, title and axes ticks
					ax.set_ylabel(yaxis_array[x], fontsize=8, color='black', family='serif')
					#ax.set_xlabel('Benchmarks', fontsize=10, color='black')
					#ax.set_title('DRAM Cache Configurations Logs\n')
					ax.set_ylim([ymin,ymax])
					xticks = ind + (0.0025 * (len(file_names)))
					ax.set_xticks(xticks)
					ax.set_xticklabels(([axis for axis in x_axis]), fontsize=6, family='serif', rotation = parameters_x_axis_rot[x])#
					#ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
					ax.ticklabel_format(family='serif')

					# Shrink current axis by 20%
					box = ax.get_position()
					ax.set_position([box.x0, box.y0, box.width * 1.3, box.height])
					# Hide the right and top spines
					ax.spines['right'].set_visible(True)
					ax.spines['top'].set_visible(True)
					ax.spines['bottom'].set_linewidth(0.5)
					ax.spines['left'].set_linewidth(0.5)
					ax.spines['right'].set_linewidth(0.5)
					ax.spines['top'].set_linewidth(0.5)

					ax.spines['bottom'].set_color('#848484')
					ax.spines['left'].set_color('#848484') 
					ax.spines['right'].set_color('#848484')
					ax.spines['top'].set_color('#848484') 
			
					# Only show ticks on the left and bottom spines
					ax.yaxis.set_ticks_position('left')
					ax.xaxis.set_ticks_position('bottom')
					# Try to set the tick markers to extend outward from the axes, R-style
					for line in ax.get_xticklines():
					    line.set_marker(mpllines.TICKDOWN)
					    line.set_markeredgewidth(0.5)
					    line.set_markeredgecolor('#848484')

					for line in ax.get_yticklines():
					    line.set_marker(mpllines.TICKLEFT)
					    line.set_markeredgewidth(0.5)
					    line.set_markeredgecolor('#848484') 
			
					ticks_font = font_manager.FontProperties(family='serif', style='normal',size=6, weight='normal', stretch='normal')
					for label in ax.get_yticklabels():
		    			    label.set_fontproperties(ticks_font)
		
					L = ax.legend([leg for leg in legends_names_array], loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True,  fontsize=6) #ncol=len(file_names)/2,shadow=True,
					L.get_frame().set_edgecolor('#848484')
					L.get_frame().set_linewidth(0.5)
					plt.setp(L.texts, family='serif')
					ax.set_axisbelow(True)

					ax.yaxis.grid(color='gray', linestyle='-', linewidth=.5)
					plt.tight_layout()
					#ax.legend( [leg for leg in legends_names_array], bbox_to_anchor=(1.04,1), loc="best")	    
					#ax.axvspan(9.5, 9.01, facecolor='g', alpha=0.5)
					#ax.axhspan(9.5, 9.01, facecolor='g', alpha=0.5)
			
					plt.axvline(x=N - 1.5, color='gray', linestyle='--', linewidth=.5)
					ax.margins(0.04)
					#ax2 = ax.twinx()
					#ax2.plot([4491, 10579, 6400], 'x-', markeredgewidth=2, color='#7e7e7e', linewidth=2, markersize=8)

					#print ([max_sl for max_sl in splitted_list])
					pdf.savefig()  # saves the current figure into a pdf page
					plt.savefig('Graphs/' + yaxis_array[x] + '_line.pdf')
					#plt.show()



#------------------------------------------------------------------------------------------------------------------#/net/home/tahmad/Documents/Python/Script/
#------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------------------------------------------#
def main(argv):
    # My main code here
    pass
    
    #Reading parameters files
    read_parameters_file()
    read_legends_file()
    read_format_file()
    read_axis_file()

    #if in case reading data from log files
    if not sys.argv[2] == 'direct':
	    #Data extraction from log files
	    data_extraction_logfiles()

    #Graphs generation
    graphs_generation()

if __name__ == "__main__":
    main(sys.argv)
#------------------------------------------------------------------------------------------------------------------#









