def summary(dev, org, count):
    arr = list(enumerate([abs(dev[i]-org[i]) for i in range(count)]))
    summ,d,o = 0,0,0
    arr2 = sorted(arr, key=lambda x: x[1])
    print('sorted', arr2)
    while d*2<count and o*2<count and d+o<count:
        index = arr2[-max(d+o,1)][0]
        if dev[index]>org[index]:
            d+=1
            summ+=dev[index]
        else:
            o+=1
            summ+=org[index]
    print(index, d+o,d,o)
    while d+o<count:
        index = arr2[-max(d+o,1)][0]
        if d<o:
            summ+=dev[index]
            d+=1
        else:
            summ+=org[index]
            o+=1
    return summ

with open('input.txt', 'r') as f1:
    partipicians = int(f1.readline())
    dev_skill = list(map(lambda x: int(x), f1.readline().split()))
    org_skill = list(map(lambda x: int(x), f1.readline().split()))
    serfs = int(f1.readline())
    ans = []
    for i in range(serfs):
        skill = list(map(lambda x: int(x), f1.readline().split()))
        if skill[1]==1:
            dev_skill[skill[0]-1]+=skill[2]
        else:
            org_skill[skill[0]-1]+=skill[2]
        ans.append(summary(dev_skill, org_skill, partipicians))
print(ans)
with open('output.txt', 'w') as f2:
    for i in ans:
        f2.write(str(i)+'\n')