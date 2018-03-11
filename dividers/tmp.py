def check(events, new_ev):
    for i in events:
        if i[0] == new_ev[0]:
            if i[1]<new_ev[1]<i[1]+i[2] or i[1]<new_ev[1]+new_ev[2]<i[1]+i[2]:
                return False
    return True


def finder(events, day, name):
    arr = []
    for i in events:
        if i[0] == day and name in i[4::]:
            arr.append(i[1::])
    arr.sort()
    for i, j in enumerate(arr):
        hours = str(j[0] // 60) if j[0]//60>9 else '0'+str(j[0]//60)
        mins = str(j[0] % 60) if j[0]%60>9 else '0'+str(j[0]%60)
        arr[i][0] = hours + ':' + mins
        arr[i][1] = str(j[1])
    return arr


with open('input.txt', 'r') as f1:
    n = int(f1.readline())
    arr = []
    answers = []
    for i in range(n):
        line = f1.readline().split()
        line[1] = int(line[1])
        if line[0] == 'APPOINT':
            time = line[2].split(':')
            line[2] = int(time[0]) * 60 + int(time[1])
            line[3] = int(line[3])
            checker = check(arr, line[1:4:])
            if checker:
                answers.append('OK')
                arr.append(line[1::])
            else:
                answers.append('FAIL')
                answers.append(line[5::])
        else:
            info = finder(arr, line[1], line[2])
            for j in info:
                j.pop(2)
                answers.append(j)

with open('output1.txt', 'w') as f2:
    for i in answers:
        if type(i)!=list:
            f2.write(i+'\n')
        else:
            f2.write(' '.join(i) + '\n')