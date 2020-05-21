#!/usr/bin/env python
#!encoding:utf-8
"""
打印不同参数组的训练曲线，以便寻找最优参数
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f_in = sys.argv[1]

def load_data():
    buff = []
    ret = {}    # {flag_depth: [(acc_train, acc_test)]}
    flag = None
    for line in open(f_in, 'r').readlines():
        if 'max_depth' in line:
            if buff:
                ret[flag] = np.array(buff)
                print("depth=%s:\t%s" % (flag, len(buff)))
                buff.clear()
            d = eval(line.strip())
            flag = d['max_depth']
            continue
        if 'N-rmse' not in line:
            continue
        seps = line.strip().split()
        assert len(seps) == 9
        rmse_0 = float(seps[1].split(':')[1])
        rmse_1 = float(seps[4].split(':')[1])
        auc_0  = float(seps[2].split(':')[1])
        auc_1  = float(seps[5].split(':')[1])
        acc_0  = float(seps[-2].split(':')[1])
        acc_1  = float(seps[-1].split(':')[1])
        buff.append((rmse_0, rmse_1, auc_0, auc_1, acc_0, acc_1))
    if buff:
        ret[flag] = np.array(buff)
        print("depth=%s:\t%s" % (flag, len(buff)))
    print("#load data: %s" % len(ret))
    return sorted(ret.items(), key = lambda x: x[0])

def plot_oneY(arr_acc):
    print("Best values in test-set:")
    for flag, accs in arr_acc:
        print("\tdepth=%2d\tacc=%.3f (%2d) \trmse=%.3f (%2d) \tAUC=%.3f (%2d)" 
         % (flag, accs[:,5].max(), accs[:,5].argmax(), accs[:,1].min(), accs[:,1].argmin(), accs[:,3].max(), accs[:,3].argmax()))
    plot_one_type(arr_acc, 0, 1, 'rmse')
    plot_one_type(arr_acc, 2, 3, 'auc')
    plot_one_type(arr_acc, 4, 5, 'acc')

def plot_one_type(arr_acc, c1, c2, vtype):
    plt.cla()   # 清空之前的数据
    x = range(arr_acc[0][1][:,0].size)
    y_min, y_max = sys.float_info.max, sys.float_info.min
    for flag, accs in arr_acc:
        plt.plot(x, accs[:,c1], label='train_%s' % flag)
        plt.plot(x, accs[:,c2], label='test__%s' % flag)
        y_min = min(accs[:,c1].min(), accs[:,c2].min(), y_min)
        y_max = max(accs[:,c1].max(), accs[:,c2].max(), y_max)
    plt.xticks(range(0, len(x), len(x)//20))                        # 设置x轴显示间隔
    plt.yticks((y_max-y_min) / 19. * np.arange(20) + y_min)         # 设置x轴显示间隔
    plt.margins(0)                                                  # 图像与坐标轴线不留空
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)   # 调节边距空白
    
    plt.title("train curve: " + vtype.upper())
    plt.xlabel("Iteration")
    plt.ylabel(vtype.title())
    plt.legend(loc=1)
    plt.grid(True, alpha=0.5)
    # plt.show()
    
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    fig.savefig(f_in + '.' + vtype + '.png', dpi=100)

arr_acc = load_data()
plot_oneY(arr_acc)

