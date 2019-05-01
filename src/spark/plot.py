import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_loss_train(fhead):
    plt.plot(iter_list, err_train_list)
    plt.xlabel('epoch')
    plt.ylabel('train_err')
    plt.title(fhead)
    plt.savefig(fhead+'train_err.png')
    #plt.close()
    #plt.show()

def plot_loss_test(fhead):
    plt.plot(iter_list, err_test_list)
    plt.xlabel('epoch')
    plt.ylabel('test_err')
    plt.title(fhead)
    plt.savefig(fhead+'test_err.png')
    #plt.show()
    #plt.close()

def save_file(head):
    np.save(head+"iter_list.npy", iter_list)
    np.save(head+"err_train_list.npy", err_train_list)
    np.save(head+"err_test_list.npy", err_test_list)
    np.save(head+"time_list.npy", time_list)

def load_file(head):
    iter_list = np.load(head+"iter_list.npy")
    err_train_list = np.load(head+"err_train_list.npy")
    err_test_list = np.load(head+"err_test_list.npy")
    return iter_list, err_train_list, err_test_list

dataset_type = "10M"
inputV_filepath = None
test_inputV_filepath = None

if dataset_type == '100k':
    inputV_filepath = "/home/jingguaz/FinalProject/data/ml-100k/ua.base"#"./nf_subsample.csv"
    test_inputV_filepath = "/home/jingguaz/FinalProject/data/ml-100k/ua.test"
elif dataset_type == '10M':
    inputV_filepath = "/home/jingguaz/sgd/movielens/data/train_shuffle.dat"
    test_inputV_filepath = "/home/jingguaz/sgd/movielens/data/test_shuffle.dat"
elif dataset_type == '100M':
    inputV_filepath = "/home/jingguaz/sgd/movielens/netflix/NetflixRatings_train.csv"
    test_inputV_filepath = "/home/jingguaz/sgd/movielens/netflix/NetflixRatings_test.csv"
num_workers = 8#16

fhead = "aws_"+dataset_type + "_" + str(num_workers) + "workers_"
iter_list, err_train_list, err_test_list = load_file(fhead)
plot_loss_train(fhead)
plot_loss_test(fhead)
