# coding:utf-8
def charu(alist):
    for i in range(1, len(alist)+1):
        for j in range(i, 1, -1):
            if alist[j] < alist[j-1]:
                alist[j], alist[j-1] = alist[j-1], alist[j]