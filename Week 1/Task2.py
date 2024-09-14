def average2d(arr2d):
    sum = 0
    for arr in arr2d:
        for el in arr:
            sum += el
    
    return sum / (len(arr2d) * len(arr2d[0]))

def sampleVariance2d(arr2d):
    avg = average2d(arr2d)
    sum = 0
    for arr in arr2d:
        for el in arr:
            sum += (el - avg) ** 2

    return sum / (len(arr2d) * len(arr2d[0]) - 1)

l2 = [[70, 13, 46, 71], [88, 89, 7, 19], [62, 95, 24, 28], [47, 15, 64, 27]]
print(average2d(l2))
print(sampleVariance2d(l2))