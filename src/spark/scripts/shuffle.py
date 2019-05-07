import random

if __name__ == '__main__':
  random.seed(20190424)

  maxUser, maxItem = 0, 0
  train_data = []
  with open('../data/train.dat', 'r') as f:
    for line in f.readlines():
      d = line.split('::')
      user, item, rating, ts = int(d[0]), int(d[1]), d[2], d[3]
      maxUser, maxItem = max(maxUser, user), max(maxItem, item)
      train_data.append((user, item, rating, ts))
  
  test_data = []
  with open('../data/test.dat', 'r') as f:
    for line in f.readlines():
      d = line.split('::')
      user, item, rating, ts = int(d[0]), int(d[1]), d[2], d[3]
      test_data.append((user, item, rating, ts)) 
  
  nUser = maxUser
  nItem = maxItem

  user_idx = list(range(1, nUser+1))
  item_idx = list(range(1, nItem+1))

  random.shuffle(user_idx)
  random.shuffle(item_idx)

  for idx in range(len(train_data)):
    user, item, rating, ts = train_data[idx]
    user = user_idx[user-1]
    item = item_idx[item-1]
    train_data[idx] = (user, item, rating, ts)
  
  for idx in range(len(test_data)):
    user, item, rating, ts = test_data[idx]
    user = user_idx[user-1]
    item = item_idx[item-1]
    test_data[idx] = (user, item, rating, ts)

  sort_func = lambda x, y: x[0] < y[0] or (x[0] == y[0] and x[1] < y[1])
  train_data.sort()
  test_data.sort()
  
  with open('../data/train_shuffle.dat', 'w') as f:
    for user, item, rating, ts in train_data:
      data_str = str(user) + "::" + str(item) + "::" + rating + "::" + ts
      f.write(data_str)
  
  with open('../data/test_shuffle.dat', 'w') as f:
    for user, item, rating, ts in test_data:
      data_str = str(user) + "::" + str(item) + "::" + rating + "::" + ts
      f.write(data_str)

