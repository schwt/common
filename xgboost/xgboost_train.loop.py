#!/data1/anaconda3/bin/python
#!encoding:utf-8
"""
遍历max_depth参数，方便调优
"""
import time
import xgboost as xgb
from Loger import Loger
import numpy as np
import sys

eta       = 0.1

path_data      = "./data/"
file_col_names = "./col_names.txt"
file_train  = path_data + "train.csv"
file_tests  = path_data + "test.csv"
path_model  = '../data/model/'
loger = Loger()
log = loger.log

def load_col_names():
    ret = []
    rf = open(file_col_names,'r')
    for line in rf.readlines():
        ret.append(line.strip().split()[1])
    log("#cols: %s" % len(ret))
    rf.close()
    return ret
        
# 正负样本准确率平均
def acc_avg(preds, dtrain):
    np_label = np.array(dtrain.get_label())
    num_pos = np_label.sum()
    np_pred_right = ((np.array(preds) >= 0.5) == np_label)
    accuracy_0  = (np_pred_right * (1-np_label)).sum() / (np_label.size - num_pos)
    accuracy_1  = (np_pred_right * np_label).sum() / num_pos
    accuracy_avg = 50 * (accuracy_0 + accuracy_1)
    return 'mAcc', round(accuracy_avg, 3)
def acc_neg(preds, dtrain):
    np_label = np.array(dtrain.get_label())
    num_pos = np_label.sum()
    np_pred_right = ((np.array(preds) >= 0.5) == np_label)
    accuracy_1 = 100. * (np_pred_right * np_label).sum() / num_pos
    return 'Acc+', accuracy_1

# 计算全部、正、负样本中的准确率
def calc_accuracy(labels, preds, if_print=False):
    total = len(labels)
    assert len(preds) == total
    np_label = np.array(labels)
    num_pos = np_label.sum()
    np_pred_right = ((np.array(preds) >= 0.5) == np_label)

    accuracy_all = 100. * np_pred_right.sum() / total
    accuracy_1   = 100. * (np_pred_right * np_label).sum() / num_pos
    accuracy_0   = 100. * (np_pred_right * (1-np_label)).sum() / (total - num_pos)
    return accuracy_all, accuracy_1, accuracy_0

def load_DMatrix(file_name, col_names, s_name):
    log("load %s ..." % file_name)
    dmatrix = xgb.DMatrix(file_name + "?format=csv&label_column=0", feature_names = col_names)
    num_1 = sum(dmatrix.get_label())
    num_0 = len(dmatrix.get_label()) - num_1
    log("# %s set: %s" % (s_name, dmatrix.num_row()))
    log("# %s   +: (%5.2f%%)   %d" % (s_name, 100. * num_1/dmatrix.num_row(), num_1))
    log("# %s   -: (%5.2f%%)   %d" % (s_name, 100. * num_0/dmatrix.num_row(), num_0))
    log("# %s -/+: %.2f"           % (s_name, 1. * num_0 / num_1))
    return dmatrix

def output_predict(bst, dtrain, f_output):
    preds  = bst.predict(dtrain)
    labels = dtrain.get_label()
    """
    wf = open(f_output, 'w')
    for i in range(len(labels)):
        wf.write("%s\t%s\t%d\n" % (int(labels[i]), preds[i], int(preds[i]>0.5) == labels[i]))
    wf.close()
    wf = open(f_output, 'w')
    """
    accuracy_all, accuracy_1, accuracy_0 = calc_accuracy(labels, preds)
    log("  accuracy   +: %.3f%%" % accuracy_1)
    log("  accuracy   -: %.3f%%" % accuracy_0)
    log("  accuracy avg: %.3f%%" % ((accuracy_1 + accuracy_0)/2))
    log("  accuracy ALL: %.3f%%" % accuracy_all)

def main():
    cols = load_col_names()
    dtrain = load_DMatrix(file_train, cols, 'train')
    dtests = load_DMatrix(file_tests, cols, 'tests')
    
    # 遍历max_depth 参数范围
    for depth in range(7, 14):
        iTrain(dtrain, dtests, depth, 50)

def iTrain(dtrain, dtests, max_depth, num_round):
    param = {'booster':'gbtree','max_depth':max_depth, 'eta':eta, 'silent':1, 'objective':'binary:logistic', 'nthread': 15}
    param['eval_metric'] = ['rmse', 'auc', 'error']
    param['scale_pos_weight'] = 2
    evallist = [(dtrain, 'N'),  (dtests, 'E')]
    log("param:\n\t" + str(param) + "\n\tnum_round: " + str(num_round))
    
    bst = xgb.train(param, dtrain, num_round, evallist, feval=acc_avg)
    loger.logu("train done.")
    file_model  = path_model + 'model.%s.%s' % (max_depth, num_round)
    file_fscore = file_model + ".score"
    bst.save_model(file_model + '.dat')
    bst.dump_model(file_model + '.txt')
    
    list_score = sorted(bst.get_score(importance_type='gain').items(), key=lambda x: -x[1])
    with open(file_fscore, 'w') as wf:
        for k,v in list_score:
            wf.write("%12.5f\t%s\n" % (v,k))
    
    log("predict train:")
    output_predict(bst, dtrain,  path_data + '/pred.train.txt.%s.%s' % (max_depth, num_round))
    log("predict test:")
    output_predict(bst, dtests,  path_data + '/pred.tests.txt.%s.%s' % (max_depth, num_round))

    
if __name__ == "__main__":
    loger0 = Loger()
    loger0.log("begin...")

    main()

    loger0.logu("done")
