import numpy as np
import findspark
findspark.init("/home/jingguaz/jingguang/bin/spark")
import pyspark
from pyspark import SparkConf, SparkContext

conf = SparkConf()
#conf.set("spark.executor.memory", "6G")
#conf.set("spark.driver.memory", "2G")
#conf.set("spark.executor.cores", "4")
#conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

num_workers = 9
num_factors = 100
num_iterations = 50
beta = 0.8
lamda = 1.0
inputV_filepath = "/home/jingguaz/FinalProject/data/ml-100k/ua.base"#"./nf_subsample.csv"
outputW_filepath = "./W.csv"
outputH_filepath = "./H.csv"

conf.set("spark.default.parallelism", str(num_workers))
conf.setMaster('local[{}]'.format(num_workers))

sc = SparkContext(conf=conf)

lines = sc.textFile(inputV_filepath)
lines = lines.map(lambda l: (l.split("\t")[:-1])) #movieslens
movies = lines.map(lambda l: int(l[1]) - 1)
users = lines.map(lambda l: int(l[0]) - 1)
scores = lines.map(lambda l: float(l[2]))
print(users.stats())
print(movies.stats())
print(scores.stats())

rows = users.max()+1
cols = movies.max()+1
V = np.zeros((rows, cols), np.float)
for tup in lines.collect():
    V[int(tup[0])-1][int(tup[1])-1] = float(tup[2])
print((V!=0).sum() / (V.shape[0] * V.shape[1]))
W =  np.random.normal(0, 0.5, (rows, num_factors))#np.random.rand(rows, num_factors) # will have nan, -8e+95
H =  np.random.normal(0, 0.5, (num_factors, cols)) #np.random.rand(num_factors, cols)
strata = np.random.permutation(num_workers)

print(W.shape, H.shape)
print(V.shape)
print(strata)
W_subs = np.array_split(W, num_workers)
H_subs = np.array_split(H, num_workers, axis=1)
print(len(W_subs), len(H_subs), W_subs[0].shape, H_subs[0].shape)
V_subs_ = np.array_split(V, num_workers)

def updateParameter(tup):
    idx, mat = tup[0], tup[1]
    W_sub = tup[2]#W_subs[idx]
    H_sub = tup[3]#H_subs[idx]
    W_sub_grad = np.zeros_like(W_sub)
    H_sub_grad = np.zeros_like(H_sub)
    
    spIdx = np.nonzero(mat)
    spLen = spIdx[0].shape[0]
    samplingCnt = 3
    #print(spIdx, spLen, idx, mat.sum())
    if (spLen == 0):
        return idx, (W_sub_grad, H_sub_grad)
    sampling = np.random.choice(spLen, samplingCnt)
    lr = 0.01
    lamda = 1
    r, c = 0, 0
    #print(sampling)
    for i in sampling:
        r, c = spIdx[0][i], spIdx[1][i]
        #print(r, c, spLen)
        tmp = np.dot(W_sub[r, :], H_sub[:, c])
        inner = mat[r][c] - tmp
        W_sub_grad[r, :] -= lr * (-2.0*(inner)*H_sub[:, c] + 
                                 2.0 * lamda * W_sub[r, :].T / W_sub.shape[1]).T
        H_sub_grad[:, c] -= lr * (-2.0*(inner)*W_sub[r, :].T +
                               2.0 * lamda * H_sub[:, c] / H_sub.shape[0])
    #return idx, (W_sub_grad, H_sub_grad), (W_sub_grad.sum(), H_sub_grad.sum(), W_sub.shape, H_sub.shape, spLen, c, W_sub[r, :])
    return idx, (W_sub_grad, H_sub_grad)

def train_one():
    strata = np.random.permutation(num_workers)
    #V_subs = sc.parallelize(zip(np.arange(num_workers), 
    #                        np.array_split(V, num_workers), W_subs, H_subs))
    input_all = []
    for i in range(num_workers):
        input_all.append((i, V_subs_[i], W_subs[i], H_subs[strata[i]]))
    V_subs = sc.parallelize(input_all)

    V_subs_subs = V_subs.map(
    lambda x: (x[0], 
               np.array_split(x[1], num_workers, axis=1)[strata[x[0]]], x[2], x[3]))
    
    tup = V_subs_subs.map(updateParameter)
    tup = tup.collect()
    for i in range(len(tup)):
        W_subs[tup[i][0]] += tup[i][1][0]
        H_subs[strata[tup[i][0]]] += tup[i][1][1]
    print("Total Loss:", (V - np.dot(np.concatenate(W_subs, axis=0), np.concatenate(H_subs, axis=1))).sum())
    

for i in range(1000):
    train_one()