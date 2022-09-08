from operator import itemgetter;
from math import sin, cos, sqrt, atan2, radians;
import time;
import math;
import sys;
import multiprocessing;

# specify the minute and space (by meters) threshold, i.e., Delta_T and Delta_S

minute = int(sys.argv[1]);
space = int(sys.argv[2]);

MAX_SPACE_INTERVAL = space;
MIN_TIME_INTERVAL = minute * 60;
SPLIT = 0.001;

MAX_STAY_TRIP_SIZE = 10000;

#radius of the earth by km
RADIUS_EARTH = 6371;
DEGREE_TO_RADIAN = 2 * math.pi / 360;
COS_LATITUDE = 0.77;

#read file

filelist = ['P2-part-0169' + str(i) for i in range(1)]

baseReadPath = "/datahouse/yurl/TalkingData/data/BJ_cleaned_data/"
baseWritePath = "/datahouse/yurl/TalkingData/data/P3-SS/"

# baseReadPath = "/home/shil/TD-exp/BJ_cleaned_data/"
# baseWritePath = "/home/shil/TD-exp/P3-SS/"

# convert seconds to hour
def convert_to_hour(seconds):
	# hour = int((seconds - 1467000000 - int((seconds - 1467000000) / (24 * 7 * 3600)) * (24 * 7 * 3600)) / 3600)
	hour = int((seconds - 1467000000)) / 3600 % (7*24)
	return hour

# convert longitude/latitude to one-hot
def convert_longitude(data, split):
	return int((data - 115.422) / split)

def convert_latitude(data, split):
	return int((data - 39.445) / split)

def distance(lat1, lon1, lat2, lon2):
	lat1 = lat1 * DEGREE_TO_RADIAN;
	lon1 = lon1 * DEGREE_TO_RADIAN;
	lat2 = lat2 * DEGREE_TO_RADIAN;
	lon2 = lon2 * DEGREE_TO_RADIAN;
	x = (lon2 - lon1) * COS_LATITUDE;
	y = lat2 - lat1;
	return int(RADIUS_EARTH * sqrt(x * x + y * y) * 1000); 

# label the file with a name of "filename"; append 0 after the stay record, append 1 after the travel record, do nothing for other records
def label(filename):
	start_time = time.time();
	filename_r = baseReadPath + filename;
	filename_w = baseWritePath + filename + "_" + str(minute) + "-" + str(space);
	content = file(filename_r, 'r').read();
	records = content.split("\n");
	# print "File" + str(file_index) + ": " + str(len(records)) + " records";
	
	with open(filename_w, 'w') as ofile:	
		ofile.write('');

	record_num = len(records);
	stay_num = 0;
	travel_num = 0;
	
	c_uid = -1;
	time_second_array = [];
	longitude_array = [];
	latitude_array = [];
	trajectory_seg_array = [];

	# iterate all records
	for record_index in range(len(records)):
		record = records[record_index];
		
		columns = record.split(",");
		
		if (len(columns)<4):
			if (len(columns)>0):
				print("an error line in line: "+str(record_index));
			continue;

		# set record columns
		uid = columns[0].strip();
		time_second = int(columns[1].strip()[0:10]);
		latitude = float(columns[2].strip());
		longitude = float(columns[3].strip());

		# check if it is the same trajectory
		if (uid == c_uid):
			# same trajectory, save into the arrays
			time_second_array.append(time_second);
			longitude_array.append(longitude);
			latitude_array.append(latitude);
		else:
			# new uid
			if c_uid != -1:
				
				# the current uid is valid, segment the trajectory of the current uid (c_uid)
				
				# sort the trajectory by time
				index_array = [i[0] for i in sorted(enumerate(time_second_array), key=itemgetter(1))];

				# truncate the trajectory into segments at every time interval larger than Delta_T, stored in trajectory_seg_array
				# the first index of the current segment
				first_index = 0;

				for next_index in range(1, len(index_array)):
					tim1 = time_second_array[index_array[next_index-1]];
					tim2 = time_second_array[index_array[next_index]];
					time_interval = tim2 - tim1;
					
					if (time_interval > MIN_TIME_INTERVAL):
						temp_seg = []
						for xindex in xrange(first_index, next_index):
							record_line = [c_uid, time_second_array[index_array[xindex]], latitude_array[index_array[xindex]], longitude_array[index_array[xindex]]]
							temp_seg.append(record_line)
						
						trajectory_seg_array.append(temp_seg)
						first_index = next_index
				
				temp_seg = []
				for xindex in xrange(first_index, len(index_array)):
					record_line = [c_uid, time_second_array[index_array[xindex]], latitude_array[index_array[xindex]], longitude_array[index_array[xindex]]]
					temp_seg.append(record_line)
				trajectory_seg_array.append(temp_seg)
			
			# refresh the arrays to only store the first record of the new trajectory (uid)
			time_second_array = [];
			longitude_array = [];
			latitude_array = [];
			time_second_array.append(time_second);
			longitude_array.append(longitude);
			latitude_array.append(latitude);
			c_uid = uid;
	
	
	print("Read and segment file " + str(filename) + " in " + str(time.time() - start_time) + " seconds...")
	start_time = time.time();
	
	
	# apply the ss algorithm on each segment from all the trajectories
	
	for seg in trajectory_seg_array:
		
		seg_record_array = [x for x in seg];
		
		# the segment with less than three records can not be labeled by our algorithm
		if len(seg_record_array) < 3:
			
			# return trip
			with open(filename_w, 'a') as ofile:
				for segment_record in seg_record_array:
				
					if(len(segment_record)<0):
						continue;
				
					ofile.write(segment_record[0]);
				
					for field_index in range(1, len(segment_record)):
						ofile.write(',' + str(segment_record[field_index]));
						
					ofile.write('\n');
	
			continue;
		
		head_index = 0;
		
		# label stay trips in the segment
		# the below algorithm according to the Algorithm 2 in the paper in ShareLatex
		for cursor_index in xrange(1, len(seg_record_array)):
			
			# too-long stay trip, cut here
			if ((cursor_index - head_index) > MAX_STAY_TRIP_SIZE):
				
				print ('Cut too-long stay trip at segment offset: %d'%(cursor_index));
				
				if seg_record_array[cursor_index-1][1] - seg_record_array[head_index][1] >= MIN_TIME_INTERVAL:
					
					for k in xrange(head_index, cursor_index):
						# only label the record not labeled as stay any more
						if len(seg_record_array[k]) == 4:
							seg_record_array[k].append(0);
							stay_num = stay_num + 1;
				
				head_index = cursor_index;
				continue;
				
			for anchor_index in xrange(cursor_index-1, head_index-1,-1):
			
				space_interval = distance(seg_record_array[cursor_index][2], seg_record_array[cursor_index][3], seg_record_array[anchor_index][2], seg_record_array[anchor_index][3])
				
				if space_interval > MAX_SPACE_INTERVAL:
					
					if seg_record_array[cursor_index-1][1] - seg_record_array[head_index][1] >= MIN_TIME_INTERVAL:
						
						for k in xrange(head_index, cursor_index):
							# only label the record not labeled as stay any more
							if len(seg_record_array[k]) == 4:
								seg_record_array[k].append(0);
								stay_num = stay_num + 1;
								
					head_index = anchor_index + 1;
					break;
		
		# handle the remaining records in the segment		
		if seg_record_array[len(seg_record_array)-1][1] - seg_record_array[head_index][1] >= MIN_TIME_INTERVAL:
			
			for k in xrange(head_index, len(seg_record_array)):
				# only label the record not labeled as stay any more
				if len(seg_record_array[k]) == 4:
					seg_record_array[k].append(0);
					stay_num = stay_num + 1;
		
		# label travel records in the segment
		# the below algorithm according to the Algorithm 2 in the paper in ShareLatex
		for cursor_index in xrange(1, len(seg_record_array) - 1):
			
			# for all the unlabeled records till now
			if len(seg_record_array[cursor_index]) == 4:
				
				left = -1;
				right = -1;
				
				# find the first out-of-range record on the left of cursor_index
				for l_index in xrange(cursor_index - 1, -1, -1):
					
					d = distance(seg_record_array[cursor_index][2], seg_record_array[cursor_index][3], seg_record_array[l_index][2], seg_record_array[l_index][3])
					
					if d > MAX_SPACE_INTERVAL:
						left = l_index;
						break;
					
					t = seg_record_array[cursor_index][1] - seg_record_array[l_index][1];
 					
					if t > MIN_TIME_INTERVAL:
						break;
								
				# find the first out-of-range record on the right of cursor_index
				for r_index in xrange(cursor_index + 1, len(seg_record_array)):
					
					d = distance(seg_record_array[cursor_index][2], seg_record_array[cursor_index][3], seg_record_array[r_index][2], seg_record_array[r_index][3])
					
					if d > MAX_SPACE_INTERVAL:
						right = r_index;
						break;
					
					t = seg_record_array[r_index][1] - seg_record_array[cursor_index][1];
 					
					if t > MIN_TIME_INTERVAL:
						break;
							
				if right!= -1 and left!= -1 and (seg_record_array[right][1] - seg_record_array[left][1] <= MIN_TIME_INTERVAL):
					seg_record_array[cursor_index].append(1);
					travel_num = travel_num + 1;
		
		# return trip
		with open(filename_w, 'a') as ofile:
			for segment_record in seg_record_array:
				
				if(len(segment_record)<0):
					continue;
				
				ofile.write(segment_record[0]);
				
				for field_index in range(1, len(segment_record)):
					ofile.write(',' + str(segment_record[field_index]));
					
				ofile.write('\n');
	
	print("Label file " + str(filename) + " in " + str(time.time() - start_time) + " seconds...");
	print("Total record: " + str(record_num) + ", Stay record: " + str(stay_num) + "(%" + '%.2f'%(100*stay_num/float(record_num)) + "), Travel record: " + str(travel_num) + "(%" + '%.2f'%(100*travel_num/float(record_num))+")");


# pool = multiprocessing.Pool(processes=15)
# pool.map(label, filelist)

for filename in filelist:
	label(filename);
