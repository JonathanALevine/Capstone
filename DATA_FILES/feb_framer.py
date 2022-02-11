import re
import datetime
import sys
import numpy as np

infile_name = sys.argv[1]
date = datetime.datetime.now()
date_string = date.strftime("%d_%m_%y")
outfile_name = 'output_data_' + date_string + '.csv'
te = str("0")
tm = str("1")

with open(infile_name, "r") as file:
    with open(outfile_name, "w") as outfile:
        count = 1
        line_1_re = "(\d.*\d*),\s*(\d*.\d*e-\d{2}),\s*(\d.*\d*),\s*(\d.*\d*)"
        line_2_lambda_re = "\((.*?),"
        line_2_trans_re = ",\s(.*?)\),"
        line_3_lambda_re = "\((.*?),"
        line_3_trans_re = ",\s(.*?)\),"
        lamb_re = "," #(1.7e-06, -0.939774),
        outfile.write("Fill Factor,Pitch,Duty Cycle,Theta,Mode,Lambda,Transmission\n")
        line_count = 0
    
        for line in file:
            if count == 1:
                line_1_match = re.match(line_1_re, line)
                if line_1_match:
                    #outfile.write(" this happened\n")
                    fill_factor = line_1_match.group(1)
                    pitch = line_1_match.group(2)
                    cycle = line_1_match.group(3)
                    depth = line_1_match.group(4)
                    
                count += 1
                continue
            if count == 2:
                line_2_l_pattern = re.compile(line_2_lambda_re)
                line_2_t_pattern = re.compile(line_2_trans_re)
                lambs = line_2_l_pattern.findall(line)
                trans = line_2_t_pattern.findall(line)
                float_list = list(np.float_(trans))
                max_trans = max(float_list, key=abs)
                max_trans_str = str(max_trans)
                max_index = float_list.index(max_trans)
                lamb_at_max = lambs[max_index]
                outline = fill_factor + ", " + pitch + ", " + cycle + ", " + depth + ", " + te + ", " + lamb_at_max + ", " + max_trans_str + "\n"  
                outfile.write(outline)
                line_count += 1
                count += 1
                continue
            if count == 3:
                line_2_l_pattern = re.compile(line_2_lambda_re)
                line_2_t_pattern = re.compile(line_2_trans_re)
                lambs = line_2_l_pattern.findall(line)
                trans = line_2_t_pattern.findall(line)
                float_list = list(np.float_(trans))
                max_trans = max(float_list, key=abs)
                max_trans_str = str(max_trans)
                max_index = float_list.index(max_trans)
                lamb_at_max = lambs[max_index]
                outline = fill_factor + ", " + pitch + ", " + cycle + ", " + depth + ", " + tm + ", " + lamb_at_max + ", " + max_trans_str + "\n"  
                outfile.write(outline)
                count += 1
                continue
            if count == 4:
                count = 1

                continue