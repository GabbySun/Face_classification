
import numpy as np
from numpy import *
import queue

#e neighborhood


class graph:
    #训练的数据，测试的数据，特征向量
    def __init__(self,train_index,train_label,val_index,val_label,calculment_file="feature.txt",e=0.6):
        self.train_index=train_index
        self.train_label=train_label
        self.val_index=val_index
        self.val_label=val_label
        self.calculment_file=calculment_file
        self.e=e
        self.similarity_computation()
        self.graph_construction()
        self.smoothness_computation()

    #计算similarity矩阵
    def similarity_computation(self):
        file=open(self.calculment_file,"r")
        features=[]
        for line in file.readlines():
            line=line.rstrip(" \n").split(" ")
            points=[float(i) for i in line]
            points=np.array(points,dtype=np.float32)
            features.append(points)
        num=len(features)
        similarity_matrix=np.zeros((num,num),dtype=np.float64)
        for i in range(num):
            for j in range(num):
                similarity_matrix[i][j]=np.linalg.norm(features[i]-features[j])
        similarity_matrix=np.exp(-similarity_matrix/mean(similarity_matrix[:,:]))
        self.similarity=similarity_matrix

    #建立图
    def graph_construction(self):
        graph_index=self.train_index
        graph_label=self.train_label
        graph_index.extend(self.val_index)
        graph_label.extend(self.val_label)
        #simi=self.similarity
        assert len(graph_index)==len(graph_label)
        num=len(graph_label)

        graphs=[]
        indicator=np.zeros(num,dtype=np.int32)
        for i in range(num):
            k = graph_index[i]
            if indicator[i] == 1:
                continue
            q = queue.PriorityQueue()
            metrix=np.zeros((958,958),dtype=np.float32)
            graph = {'index': [], 'label': []}
            q.put(k)
            graph['index'].append(k)
            graph['label'].append(graph_label[i])
            indicator[i] = 1
            while not q.empty():
                next = q.get()
                for j in range(len(graph_index)):
                    if next == graph_index[j] or indicator[j] == 1:
                        continue
                    if self.similarity[next][graph_index[j]]>=self.e:
                        q.put(graph_index[j])
                        indicator[j] = 1
                        graph['index'].append(graph_index[j])
                        graph['label'].append(graph_label[j])
                        metrix[next][graph_index[j]] = self.similarity[next][graph_index[j]]
            graph['edge_weight'] = metrix
            graphs.append(graph)

        self.graphs=graphs

    def smoothness_computation(self):
        smoothness=0
        num_of_graphs=len(self.graphs)
        for i in range(num_of_graphs):
            graph=self.graphs[i]
            index=graph['index']
            label=graph['label']
            edge_weight=graph['edge_weight']
            for j in range(len(index)):
                for k in range(j+1,len(index)):
                    a=0
                    if label[j]!=label[k]:
                        a=1

                    smoothness += 0.5 * edge_weight[j][k]*a** 2
        self.smoothness=smoothness




'''
file=open("val_index.txt","r")
index=[]
for line in file.readlines():
    line=line.rstrip("\n")
    index.append(int(line))
file2=open("val_label.txt","r")
labels=[]
for line in file2.readlines():
    line=line.rstrip("\n")
    labels.append(int(line))


print(len(index))
print(len(labels))

graphs_construction=graph("train.txt",index,labels,"feature.txt")
graphs=graphs_construction.graphs
simi=graphs_construction.similarity
print(simi)
for i in range(len(graphs)):
    graph=graphs[i]
    print(graph["index"])
print("smoothness:{}".format(graphs_construction.smoothness))


'''
