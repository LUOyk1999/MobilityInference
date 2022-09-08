def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

train_content=open('1/train-10000-30-800-1.0', 'r')
train_records=[line.rstrip('\n') for line in train_content]

for record_index in range(len(train_records)-1):
    columns=train_records[record_index].split(",")
    user=(int(columns[0]))
    time=(int(columns[1]))+3600*24*271
    lat=(float(columns[2]))-10
    log=(float(columns[3]))-10
    print_to_file('train_data_without_label',str(user)+','+str(time)+','+str(lat)+','+str(log))
    # if(len(columns) >= 5):
    #     print_to_file('test_data',str(user)+','+str(time)+','+str(lat)+','+str(log)+','+str(int(columns[4])))
    # else:
    #     print_to_file('test_data',str(user)+','+str(time)+','+str(lat)+','+str(log))