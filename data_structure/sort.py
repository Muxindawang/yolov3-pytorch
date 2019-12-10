# coding:utf-8
def bubble_sort(alist):
    """冒泡排序法"""
    n = len(alist)
    count = 0
    for j in range(n-1):
        for i in range(0, n-1-j):
            # 从头到尾
            if alist[i] > alist[i+1]:
                count += 1
                alist[i], alist[i+1] = alist[i+1], alist[i]
        if count == 0:
            break
    return alist

if __name__ == '__main__':
    li = [54, 16, 98, 31]
    print(li)
    print(bubble_sort(li))