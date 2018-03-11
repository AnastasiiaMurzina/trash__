with open('input.txt', 'r') as f1:
    n = int(f1.readline())
    words = f1.readline().split()
count = 0
r = []
for i in range(n):
    j=0
    while j<len(words[i]):
        count+=1
        j+=1
        similars = 0
        d = ''
        for k in words[:i:]:
            if (k.startswith(words[i][:j:])):
                similars+=1
                d=k
        if similars==1 and words[i]==d:
            j = len(words[i])
with open('output.txt', 'w') as f2:
    f2.write(str(count))
