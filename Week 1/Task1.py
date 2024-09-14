def average(arr):
    sum = 0

    for i in arr:
        sum += i

    return (1/len(arr) * sum)

def sampleVariance(arr):
    avg = average(arr)
    sum = 0
    for el in arr:
        sum += (el - avg) ** 2

    return sum / (len(arr) - 1)

l = [3, 5, 19, 37, 8, 113, 48, 82]
print(average(l))
print(sampleVariance(l))