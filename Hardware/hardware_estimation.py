#Author: Gokul Krishnan
#SIAM Tool

import os
import numpy as np
from subprocess import call
from pathlib import Path
from interconnect_estimation import interconnect_estimation
import time

def hardware_estimation(IN, W, weight_length, input_length ):
    if not os.path.exists('./layer_record'):
        os.makedirs('./layer_record')
    output_path = './layer_record/'
    if os.path.exists('./layer_record/trace_command.sh'):
        os.remove('./layer_record/trace_command.sh')
    f = open('./layer_record/trace_command.sh', "w")
    f.write('./SIAM/main ./SIAM/NetWork.csv '+str(weight_length)+' '+str(input_length)+' ')
    for i,(input,weight) in enumerate(zip(IN,W)):
        input_file_name = 'input_layer' + str(i) + '.csv'
        weight_file_name = 'weight_layer' + str(i) + '.csv'
        f.write(output_path + weight_file_name+' '+output_path + input_file_name+' ')
        write_matrix_weight(weight, output_path + weight_file_name)
        if len(weight.shape) > 2:
            k = weight.shape[0]
            write_matrix_activation_conv(stretch_input(input, input_length, k), None, input_length, output_path + input_file_name)
        else:
            write_matrix_activation_fc(input, input_length, None, input_length, output_path + input_file_name)
    f.close()
    call(["/bin/bash", "./layer_record/trace_command.sh"])
    start = time.time()
    interconnect_estimation()
    end = time.time()
    print("The interconnect sim time is:", (end - start))

def write_matrix_weight(input_matrix, filename):
    cout = input_matrix.shape[-1]
    weight_matrix = input_matrix.reshape(-1, cout)
    np.savetxt(filename, weight_matrix, delimiter=",", fmt='%10.5f')

def write_matrix_activation_conv(input_matrix, fill_dimension, length, filename):
    filled_matrix_b = np.zeros([input_matrix.shape[2], input_matrix.shape[1] * length], dtype=np.str)
    filled_matrix_bin, scale = dec2bin(input_matrix[0, :], length)
    for i, b in enumerate(filled_matrix_bin):
        filled_matrix_b[:, i::length] = b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",", fmt='%s')

def write_matrix_activation_fc(input_matrix, input_length, fill_dimension, length, filename):
    input_shape = input_matrix.shape
    #print(input_shape)
    my_file = Path("./to_interconnect/ip_activation.csv")
    if my_file.is_file():
        with open('./to_interconnect/ip_activation.csv', 'a') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[1]*input_length))
            f.write("\n")
            f.close()
    else:
        with open('./to_interconnect/ip_activation.csv', 'w') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[1]*input_length))
            f.write("\n")
            f.close()
    filled_matrix_b = np.zeros([input_matrix.shape[1], length], dtype=np.str)
    filled_matrix_bin, scale = dec2bin(input_matrix[0, :], length)
    for i, b in enumerate(filled_matrix_bin):
        filled_matrix_b[:, i] = b
    np.savetxt(filename, filled_matrix_b, delimiter=",", fmt='%s')



def stretch_input(input_matrix, input_length, window_size = 5):
    input_shape = input_matrix.shape
    my_file = Path("./to_interconnect/ip_activation.csv")
    if my_file.is_file():
        with open('./to_interconnect/ip_activation.csv', 'a') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[3]*input_shape[2]*input_shape[1]*input_length))
            f.write("\n")
            f.close()
    else:
        with open('./to_interconnect/ip_activation.csv', 'w') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[3]*input_shape[2]*input_shape[1]*input_length))
            f.write("\n")
            f.close()
    #print("input_shape", input_shape)
    item_num_1 = ((input_shape[2]) - window_size + 1) * ((input_shape[3])-window_size + 1)
    item_num = max(item_num_1, 1);
    if (item_num_1==0):
        output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*(window_size-1)*(window_size-1)))
    else:
        output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*(window_size)*window_size))
    iter = 0
    for i in range( max(input_shape[2]-window_size + 1, 1) ):
        for j in range( max(input_shape[3]-window_size + 1, 1) ):
            for b in range(input_shape[0]):
                if (item_num_1==0):
                    #print("Entered")
                    #print ("window size is: ", window_size)
                    output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size-1,j: j+window_size-1].reshape(input_shape[1]*(window_size-1)*(window_size-1))
                else:
                    output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size,j: j+window_size].reshape(input_shape[1]*window_size*window_size)
                #output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size,j: j+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1

    return output_matrix


def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list
