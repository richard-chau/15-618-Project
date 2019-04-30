import numpy as np
import findspark
findspark.init("/home/jingguaz/jingguang/bin/spark")
import pyspark
from pyspark import SparkConf, SparkContext

import matplotlib.pyplot as plt

conf = SparkConf()
#conf.set("spark.executor.memory", "6G")
conf.set("spark.driver.memory", '32G')#"2G")
#conf.set("spark.executor.cores", "4")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

num_workers = 24
num_factors = 100
num_iterations = 50
beta = 0.8
lamda = 1.0#0.02#1.0
samplingCnt = 100

dataset_type = "10M"
#dataset_type = "100k"
inputV_filepath = None
test_inputV_filepath = None

if dataset_type == '100k':
    inputV_filepath = "/home/jingguaz/FinalProject/data/ml-100k/ua.base"#"./nf_subsample.csv"
    test_inputV_filepath = "/home/jingguaz/FinalProject/data/ml-100k/ua.test"
elif dataset_type == '10M':
    inputV_filepath = "/home/jingguaz/sgd/movielens/data/train_shuffle.dat"
    test_inputV_filepath = "/home/jingguaz/sgd/movielens/data/test_shuffle.dat"

outputW_filepath = "./W.csv"
outputH_filepath = "./H.csv"

conf.set("spark.default.parallelism", str(num_workers))
conf.setMaster('local[{}]'.format(num_workers))

sc = SparkContext(conf=conf)


import numpy as np
import math

def parsing(line):
    if dataset_type == "100k":
        line = line.split("\t")
    if dataset_type == "10M":
        line = line.split("::")
    return [int(line[0])-1, int(line[1])-1, float(line[2])] # start from 0

lines = sc.textFile(inputV_filepath, num_workers)
lines = lines.map(lambda x: parsing(x)).persist()



numUsers = lines.max(lambda x :x[0])[0] + 1 #['943', '2', '5']
numMovies = lines.max(lambda x :x[1])[1] + 1
blockUsers = int(math.ceil(numUsers/num_workers))
blockMovies = int(math.ceil(numMovies/num_workers))
totalData = lines.count()
averRate = lines.map(lambda x: x[2]).reduce(lambda x, y: x+y)
scale = (averRate / totalData / num_factors) ** (1./2)

#H = lines.map(lambda x: (x[1], np.random.rand(num_factors,1)))
#W = lines.map(lambda x: (x[0], np.random.rand(1, num_factors)))

#(Movie, (Nj, vec))
H = lines.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[0], (x[1], np.random.normal(0, 0.5*scale, (num_factors, 1)))))

#(User, (Ni, vec))
W = lines.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[0], (x[1], np.random.normal(0, 0.5*scale, (1, num_factors)))))

                                                    
V = lines.keyBy(lambda x: int(x[0] / blockUsers)).partitionBy(num_workers).persist() #(node, (user, movie, rate))
lines.unpersist()

test_lines = sc.textFile(test_inputV_filepath).map(lambda x: parsing(x))
test_totalData = test_lines.count()
test_V = test_lines.keyBy(lambda x: int(x[0] / blockUsers)).partitionBy(num_workers).persist() #(node, (user, movie, rate))

dummy = 1
print(dummy)
def updateParameter1(tup, n):
    #tup = list(tup_iter)#(tup_iter.collect()) 
    data = tup[1]
    V_sub, W_sub, H_sub = data[0], data[1], data[2]
    #samplingCnt = V_sub.count()
    #Wdict = W_sub.collectAsMap() #{user: (N, vec)} #'ResultIterable' object has no attribute 'collectAsMap'
    #Hdict = H_sub.collectAsMap() #{movie: (N, vec)}
    #print(dict(W_sub))
    Wdict = dict(W_sub)
    Hdict = dict(H_sub)
    
    Wdict_grad = {}
    Hdict_grad = {}
    for k, v in Wdict.items():
        Wdict_grad[k] = np.zeros_like(v[1])
    for k, v in Hdict.items():
        Hdict_grad[k] = np.zeros_like(v[1])
    
    
    
    #for i in range(samplingCnt):
    err = 0
    
    V_sub = list(V_sub)
    
    if len(V_sub) != 0:
        sampling = np.random.choice(len(V_sub), 100)

        for cnt, item in enumerate(V_sub):
        #for cnt, s in enumerate(sampling):
        #    item = V_sub[s]

            u, m, r = item[0], item[1], item[2]
            Ni, Wi = Wdict[u]
            Nj, Hj = Hdict[m]
            tmp = np.dot(Wi, Hj)
            inner = (r - tmp)
            #err += inner
            lr = 0.001
            #lr = (100 + 10*n) ** (-beta)
            #lr = (100 + n + cnt) ** (-beta)
            Wi -= lr * (-2.0*(inner)*Hj + 
                                     2.0 * lamda * Wi.T / Ni).T # W_sub.shape[1]).T
            Hj -= lr * (-2.0*(inner)*Wi.T +
                                   2.0 * lamda * Hj / Nj) # H_sub.shape[0])
            Wdict[u] = (Ni, Wi)
            Hdict[m] = (Nj, Hj)
            newinner = r - np.dot(Wi, Hj)
            err += (newinner ** 2 - inner ** 2)
            
            '''
            Wdict_grad[u] -= lr * (-2.0*(inner)*Hj + 
                                     2.0 * lamda * Wi.T / Ni).T # W_sub.shape[1]).T
            Hdict_grad[m] -= lr * (-2.0*(inner)*Wi.T +
                                   2.0 * lamda * Hj / Nj) # H_sub.shape[0])
            newinner = r - np.dot(Wi + Wdict_grad[u], Hj + Hdict_grad[m])
            err += (newinner ** 2 - inner ** 2)
            
        for k, v in Wdict_grad.items():
            Wdict[k] = (Wdict[k][0], Wdict[k][1] + Wdict_grad[k])
        for k, v in Hdict_grad.items():
            Hdict[k] = (Hdict[k][0], Hdict[k][1] + Hdict_grad[k])
       ''' 
    
    err_re = 0
    '''
    for cnt, item in enumerate(V_sub):
        u, m, r = item[0], item[1], item[2]
        Ni, Wi = Wdict[u]
        Nj, Hj = Hdict[m]
        tmp = np.dot(Wi, Hj)[0][0]
        inner = r - tmp
        #err += inner#(inner**2)
        err_re += inner**2
    '''
    
    #[(user, rate)...]
    return (list(Wdict.items()), list(Hdict.items()), err, err_re) #Python3: items() now return iterators
    #return Wdict.items(), Hdict.items()
    

def computeError(tup):
    item = tup[1]
    err_sub = 0
   
    u, m, r = item[0], item[1], item[2]
    
    if u not in W_broad.value or m not in H_broad.value:
        Wi = np.random.normal(0, 0.5*scale, (num_factors, 1))
        Hj = np.random.normal(0, 0.5*scale, (1, num_factors))
        return 0
    else:
        Ni, Wi = W_broad.value[u]
        Nj, Hj = H_broad.value[m]
    tmp = np.dot(Wi, Hj)[0][0]
    inner = r - tmp
    #err_sub += inner#(inner**2)
    err_sub += inner**2
    #return err_sub ** (1./2)
    return err_sub


err_train_list = []
err_test_list = []
time_list = []
iter_list = []
unpersist_list = []

import time

#%pdb
iter_n = 100
#strata = np.random.permutation(num_workers)
#H_subs = H.keyBy( lambda x: strata[int(x[0] / blockMovies)]).partitionBy(num_workers) #[node, (movie, (N, vec))]
#W_subs = W.keyBy( lambda x: int(x[0] / blockUsers)).partitionBy(num_workers) #[node, (user, (N, vec))]

H_broad = None
W_broad = None

#def train_one(itr):
#    global H, W, H_broad, W_broad
#    global iter_n

for itr in range(1000):
    
    t_begin = time.process_time()
    strata = np.random.permutation(num_workers)
    V_subs = V.filter(lambda x: strata[int(x[1][1]/blockMovies)] == x[0]) #[node, (user, movie,  vec)]
    H_subs = H.keyBy( lambda x: strata[int(x[0] / blockMovies)])#.partitionBy(num_workers) #[node, (movie, (N, vec))] 
    #print(W.keys().collect())
    W_subs = W.keyBy( lambda x: int(x[0] / blockUsers))#.partitionBy(num_workers) #[node, (user, (N, vec))] 
    
    argsInput = V_subs.groupWith(W_subs, H_subs).partitionBy(num_workers)
    
    argsOutput = argsInput.map(lambda x: updateParameter1(x, iter_n), preservesPartitioning=True).persist() #[node, (Wdict, Hdict)]
    unpersist_list.append(argsOutput)
    #W_subs = argsOutput.map(lambda x: (x[0], x[1][0]))
    #H_subs = argsOutput.map(lambda x: (x[0], x[1][1]))
    #err = argsOutput.map(lambda x: x[1][2]).reduce(lambda x, y: x+y) #for mapPartitions
    #W = argsOutput.map(lambda x: x[0]) #[[(user, (N, vec))]*9] 
    W = argsOutput.flatMap(lambda x: x[0], preservesPartitioning=True)#.persist()
    #H = argsOutput.map(lambda x: x[1]) 
    H = argsOutput.flatMap(lambda x: x[1])#.persist()
    #print(W.collect())
    #argsOutput.unpersist()
    #print("Pass Time: total {0:.4f}".format(time.process_time() - t_begin )) 
    if itr % 10 != 0:
        continue#return
        
    t_mid = time.process_time() 
    
    err = argsOutput.map(lambda x: x[2])
    #print(err.collect())
    err = err.reduce(lambda x, y: x+y) #for map
    #err_re = argsOutput.map(lambda x: x[3]).reduce(lambda x, y: x+y)
    #t_mid1 = time.process_time() 
    
    W_broad = sc.broadcast(W.collectAsMap())
    H_broad = sc.broadcast(H.collectAsMap())
    for i in range(len(unpersist_list)):
        unpersist_list[i].unpersist() 
    #t_mid2 = time.process_time() 
    #print(V.first()) #(0, [0, 0, 5.0])
    err_total = V.map(computeError).reduce(lambda x,y: x+y)
    
    #t_mid3 = time.process_time() 
    
    err_test = test_V.map(computeError).reduce(lambda x,y: x+y)
    
    t_end = time.process_time() 
    
    #iter_n = iter_n + totalData
    iter_n = iter_n + itr
    #iter_n = iter_n + num_workers * 100
    #W_subs.collect()
    
    train_err = (err_total/totalData)** (1./2)
    test_err =  (err_test/test_totalData)** (1./2)
    err_train_list.append(train_err)
    err_test_list.append(test_err)
    time_list.append(t_mid - t_begin)
    iter_list.append(itr)
    
    print("Total Loss at Iter {}, lr = {}:".format(itr, (100 + iter_n) ** (-beta)), 0, train_err, test_err)
    #print("Pass Time: total {0:.4f}, {1:.4f}".format(t_mid - t_begin, (time.process_time() - t_begin) ) )
    #print(t_mid1 - t_mid, t_mid2 - t_mid1, t_mid3 - t_mid2, t_end - t_mid3)
    #print("Total Loss at Iter {}, lr = {}:".format(itr, (100 + iter_n) ** (-beta)), err, err_total/totalData)

    
#for i in range(1000):
#    train_one(i)
    
def plot_loss_train(fhead):
    plt.plot(iter_list, err_train_list)
    plt.xlabel('epoch')
    plt.ylabel('train_err')
    plt.title(fhead)
    plt.savefig(fhead+'train_err.png')
    plt.close()
    #plt.show()

def plot_loss_test(fhead):
    plt.plot(iter_list, err_test_list)
    plt.xlabel('epoch')
    plt.ylabel('test_err')
    plt.title(fhead)
    plt.savefig(fhead+'test_err.png')
    #plt.show()
    plt.close()
    
def save_file(head):
    np.save(head+"iter_list.npy", iter_list)
    np.save(head+"err_train_list.npy", err_train_list)
    np.save(head+"err_test_list.npy", err_test_list)
    np.save(head+"time_list.npy", time_list)

fhead = "10M_" + str(num_workers) + "workers_"
plot_loss_train(fhead)
plot_loss_test(fhead)
save_file(fhead)
