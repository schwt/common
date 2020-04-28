#!/usr/bin/env python
# -*- coding: utf8 -*-
import arrow

class Loger():
    # 打点
    def dot(self):
        self.ts = arrow.now().timestamp

    def __init__(self):
        self.dot()
        self.form = 'YYYY-MM-DD HH:mm:ss'

    def log(self, info):
        self.dot()
        str_now  = arrow.now().format(self.form)
        print("[%s] %s" % (str_now, info))


    def logu(self, info):
        ts_diff = arrow.now().timestamp - self.ts 
        str_used = arrow.get(ts_diff).format(self.form)
        self.dot()
        if ts_diff < 61:
            str_used = str_used[-2:]
        elif ts_diff < 3601:
            str_used = str_used[-5:]
        else:
            str_used = str_used[-8:]
        self.log("(%s) %s" % (str_used, info))

