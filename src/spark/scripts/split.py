#!/usr/bin/env python2
import random

ORI_DATA_DIR = '../ml-10M100K/'
NEW_DATA_DIR = '../data/'

if __name__ == '__main__':
  random.seed(20190424)

  # remove gaps in item id
  items = []
  with open(ORI_DATA_DIR+'/movies.dat', 'r') as f:
    for line in f.readlines():
      items.append(int(line.split('::')[0]))
  inv_item_idx = {}
  for idx, item in enumerate(items):
    inv_item_idx[item] = idx+1 # index starts from 1


  ratings = []
  user_idx = {} # user -> [rating indices]
  item_idx = {} # item -> [rating indices]
  with open(ORI_DATA_DIR+'/ratings.dat', 'r') as f:
    for idx, line in enumerate(f.readlines()):
      d = line.split('::')
      user, item, rating, ts = int(d[0]), int(d[1]), d[2], d[3]
      item = inv_item_idx[item]
      ratings.append((user, item, rating, ts))

      if user not in user_idx:
        user_idx[user] = []
      user_idx[user].append(idx)
      if item not in item_idx:
        item_idx[item] = []
      item_idx[item].append(idx)
    
  ratings_idx = list(range(len(ratings)))
  random.shuffle(ratings_idx)
  train_size = int(0.8 * len(ratings))
  train_idx, test_idx = set(ratings_idx[:train_size]), set(ratings_idx[train_size:])
  user_in_train = set()
  item_in_train = set()
  for idx in train_idx:
    user, item, _, _ = ratings[idx]
    user_in_train.add(user)
    item_in_train.add(item)
  
  # make sure at least one rating of each user and item is in train set
  for user, indices in user_idx.items():
    if user not in user_in_train:
      assert(len(indices) > 0)
      idx = indices[0]
      assert(ratings[idx][0] == user)
      item = ratings[idx][1]
      train_idx.add(idx)
      test_idx.remove(idx)
      user_in_train.add(user)
      item_in_train.add(item)
  
  for item, indices in item_idx.items():
    if item not in item_in_train:
      assert(len(indices) > 0)
      idx = indices[0]
      assert(ratings[idx][1] == item)
      user = ratings[idx][1]
      assert(user in user_in_train)
      train_idx.add(idx)
      test_idx.remove(idx)
      item_in_train.add(item)

  # write to file
  with open(NEW_DATA_DIR+'/train.dat', 'w') as f:
    for idx in sorted(list(train_idx)):
      user, item, rating, ts = ratings[idx]
      data_str = str(user) + "::" + str(item) + "::" + rating + "::" + ts
      f.write(data_str)

  with open(NEW_DATA_DIR+'/test.dat', 'w') as f:
    for idx in sorted(list(test_idx)):
      user, item, rating, ts = ratings[idx]
      data_str = str(user) + "::" + str(item) + "::" + rating + "::" + ts
      f.write(data_str)

