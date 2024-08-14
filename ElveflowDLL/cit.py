arr = [3, 1, 2, 1, 3, 3, 2, 4, 3, 4]

h = dict()
i = 0
target = 1
j = 1
h[arr[i]] = 1
best = 1

while j < len(arr):
    h[arr[j]] = h.setdefault(arr[j], 0) + 1
    #print(i, j, h)
    b = True
    for x in h.keys():
        if h[x] != target:
            b = False
            break
    if b:
        #print("valid", i, j)
        best = max(best, j - i + 1)
    if h[arr[j]] > target:
        
        target +=1
    
    j+=1
    if j == len(arr):
        j-=1
        i +=1
        if i == len(arr): 
            break
        h[arr[i-1]] -=1
        #print("i removed?", h)
        if h[arr[i-1]] == 0:
            h.pop(arr[i-1])
        while j > best + i-1:
            h[arr[j]] -=1
            if h[arr[j]] == 0:
                h.pop(arr[j])
            #print("j removed?", h)
            j-=1
        h[arr[j]] -=1
        #print("readjust", i, j)
        target = max(h.values())

print("ANSWER", best)





