import csv
import sys
import numpy as np
import xgboost as xgb
import datetime
import gc
reader = csv.reader(file(sys.argv[1]))
tol = []
for row in reader:
    row = [float(x) for x in row]
    tol.append(row)
tol = np.array(tol)
weight = tol[:, 0:1]
y = tol[:, 1]
X = tol[:, 2:]
print 'weight', weight 
print 'y', y
print 'X', X
dtrain = xgb.DMatrix(X, label=y, missing = -999.0, weight=weight)
del X
del y
del weight
gc.collect()
param = {}
param['eta'] = 0.02
param['max_depth'] = 5
param['eval_metric'] = 'rmse'
param['silent'] = 1
param['nthread'] = 8
plst = list(param.items())
watchlist = [ (dtrain,'train') ]
num_round = 190
start = datetime.datetime.now()
print 'begin'
bst = xgb.train(plst, dtrain, num_round, watchlist);
end = datetime.datetime.now()
print 'run time is '
print (end - start).seconds
reader = csv.reader(file(sys.argv[2]))
tol = []
for row in reader:
    row = [float(x) for x in row]
    tol.append(row)
tol = np.array(tol)
weight = tol[:, 0:1]
y = tol[:, 1]
X = tol[:, 2:]
print 'weight', weight 
print 'y', y
print 'X', X
dtest = xgb.DMatrix(X, label=y, missing = -999.0, weight=weight)
del X
del y
del weight
pred = bst.predict(dtest)
out = open(sys.argv[3], 'w')
for line in pred:
    out.write(str(line) + '\n')
out.close()
