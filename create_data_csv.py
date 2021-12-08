import csv
import os
import sys
from datetime import datetime, timedelta

import glob

import argparse
parser 	= argparse.ArgumentParser(description='')
parser.add_argument('--datetime', default="16070800", type=str, help='')
args 	= parser.parse_args()

img_ppm_dir 	= "/home/daolq/Documents/himawari8/data/raw/tc/doksuri/img"
amv_ppm_dir 	= "/home/daolq/Documents/himawari8/data/raw/tc/doksuri/amv"
f = open("bst_all.txt", "r")
Lines = f.readlines()

def generate_csv_datetime(datetime_str):
    
    yyyy 			= 2000 + int(datetime_str[0:2])
    mm 				= (int)(datetime_str[2:4])
    dd 				= (int)(datetime_str[4:6])
    hh 				= (int)(datetime_str[6:8])
    current_time 	= datetime(yyyy, mm, dd, hh, 0, 0)
    with open('train.csv', mode='w') as train_csv:
        train_writer = csv.writer(train_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open('valid.csv', mode='w') as valid_csv:
            valid_writer = csv.writer(valid_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            with open('test.csv', mode='w') as test_csv:
                test_writer = csv.writer(test_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                dem=0
                for prev in range(0, 24*90+1, 1):
                    anchor_time 	= current_time 	+ timedelta(hours=prev)

                    anchor_name 	= "{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}".format(anchor_time.year,
                                                                                anchor_time.month,
                                                                                anchor_time.day, 
                                                                                anchor_time.hour,
                                                                                anchor_time.minute)
                    if 	os.path.isfile(os.path.join(img_ppm_dir, "%s.tir.01.fld.ppm" % anchor_name)) and \
                        os.path.isfile(os.path.join(amv_ppm_dir, "%s.tir.01.fld.ppm" % anchor_name)):
                        check = False
                        for line in Lines:
                            if (line[0:5] = "66666"): dem = dem +1
                        #     x = line[0:8]
                        #     print(x, "  ",anchor_name)
                        #     if (x==anchor_name[2:10]):
                                
                        #         check = True
                        #         label = (int)(line[13:14])
                        #         center_lat = (int)(line[15:18])
                        #         center_long = (int)(line[19:23])
                        #         center_x = (int)(((600-center_lat)/1200)*443)
                        #         center_y =(int)(((center_long-850)/1200)*443)
                        #         x_min = max(0,center_x-15)
                        #         y_min = max(0,center_y-15)
                        #         x_max = min(443,center_x+15)
                        #         y_max = min(443,center_y+15)
                        #         dir_img = os.path.join(img_ppm_dir,"%s.tir.01.fld.ppm" %anchor_name)
                        #         dir_amv = os.path.join(amv_ppm_dir,"%s.tir.01.fld.ppm" %anchor_name)
                        #         pos = dem % 5
                        #         if (pos == 4):
                        #             test_writer.writerow([dir_img,dir_amv,443,443,label,x_min,y_min,x_max,y_max,label])
                        #         elif (pos==3):
                        #             valid_writer.writerow([dir_img,dir_amv,443,443,label,x_min,y_min,x_max,y_max,label])
                        #         else:
                        #             train_writer.writerow([dir_img,dir_amv,443,443,label,x_min,y_min,x_max,y_max,label])
                        # if check == False:
                        #     dir_img = os.path.join(img_ppm_dir,"%s.tir.01.fld.ppm" %anchor_name)
                        #     dir_amv = os.path.join(amv_ppm_dir,"%s.tir.01.fld.ppm" %anchor_name)
                        #     pos = prev % 5
                        #     if (pos == 4):
                        #         test_writer.writerow([dir_img,dir_amv,443,443,"","","","","",""])
                        #     elif (pos==3):
                        #         valid_writer.writerow([dir_img,dir_amv,443,443,"","","","","",""])
                        #     else:
                        #         train_writer.writerow([dir_img,dir_amv,443,443,"","","","","",""])
                print(dem)
if __name__ == '__main__':
	if args.datetime is not None:
		generate_csv_datetime(args.datetime)