#kmer
import re
import itertools
from collections import Counter
from matplotlib.cbook import print_cycles
import numpy as np
import pandas as pd

import sys

from sklearn.preprocessing import MinMaxScaler

sys.path.extend(["../../", "../", "./"])
import sys, os
import pandas as pd
import numpy as np


#将序列迭代分成含k个碱基的序列
def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

#将序列迭代分成含k个碱基的序列
def Kmer(fastas, k=2, type="RNA_features", upto=False, normalize=True, **kw):
    encoding = []
    header = ['#', 'label']
    NA = 'ACGU' #四种常见碱基，腺嘌呤（Adenine， A）、鸟嘌呤（Guanine，G）、胞嘧啶（Cytosine，C）和胸腺嘧啶（Thymine， T）
    if type in ("RNA_features", 'DNA'):
        NA = 'ACGU'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            #笛卡尔积！product(A,repeat=3)等价于product(A,A,A)
            #tmpk = range(1,k+1) -->tmpk = 1,...,k
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            print('i:',i)
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            # print('name:',name,'sequence:',sequence,'label:',label)
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [name, label]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        #意思就是先将所有的2-mer可能性先笛卡尔积出来，然后再将序列划分成含k个碱基的序列
        #用count对出现的碱基序列的次数统计，再将其值进行归一化
        #最后将值输入进开始的2-mer笛卡尔积结果，返回就完成了特征提取，结果就是文中句Qk矩阵
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        print('header:',header)
        for i in fastas:
            sequence = i.strip()
            # print('sequence:',sequence)
            kmers = kmerArray(sequence, k)
            # print('kmers:',kmers)
            count = Counter() #对元素计数
            count.update(kmers)
            # print('count:',count)
            if normalize == True: #把数值归一化成0-1的值
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    # np.savetxt("{}-mer".format(k), encoding)
    # pd.DataFrame(encoding).to_csv("{}-mer.csv".format(k), header=None, index=False)
    return np.array(encoding)

"""
    >+sample
    UUGUUAUUCUUCUUCUUUUUCUUACUCUUUCCAGUUUCCAC
    读取数据的方法可以借鉴！！！
"""
def read_fasta(file):
    f = open(file)
    documents = f.readlines()
    string = ""
    flag = 0
    fea=[]
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            string=string.upper()
            fea.append(string)
            string = ""
        else:
            string += document
            string = string.strip() #剥离空白字符
            string=string.replace(" ", "")
        # print('document:',document)
        # print('string:',string)
    fea.append(string)
    f.close()
    return fea

# def main(path):
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('-fasta', required=True, help="fasta file name")
# #     args = parser.parse_args()
# #     print(args)
#     fasta = read_fasta(path)
# # print(fasta)
#     print(np.shape(fasta))

#     # feature_name=["Kmer"]
#     # feature={"Kmer":"Kmer(fasta)"}
#     # for i in feature_name:
#     #         eval(feature[i])
#     return feature

# if __name__ == '__main__':
#     main()