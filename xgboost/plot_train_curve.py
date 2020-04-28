#!/home/admin/anaconda3/bin/python
#!encoding:utf-8
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f_in = sys.argv[1]

def load_data():
    buff_train, buff_tests = [], []
    for line in open(f_in, 'r').readlines():
        if 'R-rmse' not in line:
            continue
        seps = line.strip().split()
        assert len(seps) == 9
        acc_train = float(seps[-2].split(':')[1])
        acc_tests = float(seps[-1].split(':')[1])
        buff_train.append(acc_train)
        buff_tests.append(acc_tests)
    print("#load data: %s" % len(buff_train))
    return np.array(buff_train), np.array(buff_tests)


def plot_oneY(acc_train, acc_test):
    print("#Best: %g%% (@%d)" % (acc_test.max(), acc_test.argmax()))
    Min = min(acc_train.min(), acc_test.min())
    Max = max(acc_train.max(), acc_test.max())
    plt.ylim(Min - (Max-Min)/5, Max + (Max-Min)/5) # 设置y轴范围
    # plt.ylim(64.5, 65.5)
    x = range(acc_train.size)
    plt.plot(x, acc_train, label='acc_train')
    plt.plot(x, acc_test,  label='acc_test')
    plt.xticks(range(0, acc_train.size, acc_train.size//20)) # 设置x轴显示间隔
    plt.margins(0)                       # 图像与坐标轴线不留空
    plt.subplots_adjust(bottom=0.15)     # 调节底部空白
    
    plt.title("train curve")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy(%)")
    plt.legend(loc=2)
    plt.grid(True)
    # plt.show()
    
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    fig.savefig(f_in + '.png', dpi=100)

acc_train, acc_test = load_data()
plot_oneY(acc_train, acc_test)

