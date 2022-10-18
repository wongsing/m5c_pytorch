import pandas as pd
import numpy as np
import sys
import itertools
import os



def dnc_key(physical_chemical_path,fill_NA='0'):
    data=pd.read_csv(physical_chemical_path,header=None,index_col=None)#read the phisical chemichy proporties
    prop_key=data.values[:,0]

    if fill_NA=="1":
        prop_key[21]='NA'
    # print('prop_key:',prop_key)

    prop_data=data.values[:,1:] #第0列是碱基组合
    prop_data=np.matrix(prop_data)
    DNC_value=np.array(prop_data).T
    DNC_value_scale=[[]]*len(DNC_value)
    for i in list(range(len(DNC_value))):
        average_=sum(DNC_value[i]*1.0/len(DNC_value[i]))
        std_=np.std(DNC_value[i],ddof=1)
        DNC_value_scale[i]=[round((e-average_)/std_,2) for e in DNC_value[i]]
    prop_data_transformed=list(zip(*DNC_value_scale))
    # prop_data_transformed=StandardScaler().fit_transform(prop_data)
    #以上于pseDNC，对物化属性值进行一致处理
    return prop_data_transformed,prop_key

def get_pcp(sequence,prop_data_transformed,prop_key,LAMDA=4):
    prop_len=len(prop_data_transformed[0])

    #获取序列
    fh = open(sequence)
    seq=[]
    for line in fh:#get the fasta sequence
        if line.startswith('>'):
            pass
        else:
            seq.append(line.replace('\n','').replace('\r',''))
    fh.close()

    whole_m6a_seq=seq
    i=0
    phisical_chemichy_len=len(prop_data_transformed)#the length of properties
    sequence_line_len=len(seq[0])#the length of one sequence
    # LAMDA=4
    finally_result=[]#used to save the fanal result
    #PCP matrix 是 物化属性的归一化矩阵！
    for one_m6a_sequence_line in whole_m6a_seq:
        one_sequence_value=[[]]*(sequence_line_len-1)
        PC_m=[0.0]*prop_len
        PC_m=np.array(PC_m)
        #将物化特征值赋值给相应的序列
        for one_sequence_index in range(sequence_line_len-1):
            for prop_index in list(range(len(prop_key))):
                if one_m6a_sequence_line[one_sequence_index:one_sequence_index+2]==prop_key[prop_index]:
                    one_sequence_value[one_sequence_index]=prop_data_transformed[prop_index]
            PC_m+=np.array(one_sequence_value[one_sequence_index])
        # print('before PC_m:',PC_m)
        PC_m=PC_m/(sequence_line_len-1)
        # print('after PC_m:',PC_m)

        auto_value=[]
        for LAMDA_index in list(range(1,LAMDA+1)):
            temp = [0.0] * prop_len
            temp=np.array(temp)
            #求自协方差！
            for auto_index in list(range(1,sequence_line_len-LAMDA_index)):
                temp=temp+(np.array(one_sequence_value[auto_index-1])-PC_m)*(np.array(one_sequence_value[auto_index+LAMDA_index-1])-PC_m)
                temp=[round(e,8) for e in temp.astype(float)]
            x=[round(e/(sequence_line_len-LAMDA_index-1),8) for e in temp]
            auto_value.extend([round(e,8) for e in x])
        # print('求完自协方差auto_value:',auto_value)
        #求互协方差！
        for LAMDA_index in list(range(1, LAMDA + 1)):
            for i in list(range(1,prop_len+1)): #lamda1
                for j in list(range(1,prop_len+1)): #lamda2
                    temp2=0.0
                    if i != j:
                        for auto_index in list(range(1, sequence_line_len - LAMDA_index)):
                                temp2+=(one_sequence_value[auto_index-1][i-1]-PC_m[i-1])*(one_sequence_value[auto_index+LAMDA_index-1][j-1]-PC_m[j-1])
                        auto_value.append(round(temp2/((sequence_line_len-1)-LAMDA_index),8))
        # print('求完互协方差auto_value:',auto_value)
        finally_result.append(auto_value)
    # finally_result=MinMaxScaler().fit_transform(np.matrix(finally_result))
    print(np.array(finally_result).shape)
    # pd.DataFrame(finally_result).to_csv(r'pcp.csv',header=None,index=False)
    return np.array(finally_result)

