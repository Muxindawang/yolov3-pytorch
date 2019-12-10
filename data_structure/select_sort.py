def select_sort(alist):
    """选择排序法"""
    n = len(alist)

    for j in range(0, n-1):
        min = alist[j]
        for i in range(j+1, n):
            if alist[i] < min:
                alist[j], alist[i] = alist[i], alist[j]
    return alist


if __name__ == "__main__":
    li = [1, 5, 8, 4, 3, 2]
    print(li)
    print(select_sort(li))