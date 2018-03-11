with open('input.txt', 'r') as f1:
	k,m,d = f1.readline().split()
	k,m,d = int(k), int(m), int(d)
books_today = 1
have_books = m
#today = d
while have_books>0:
    # print('m',have_books, books_today)
    have_books -= books_today
    # print('a',have_books, books_today)
    # print(have_books)
    if have_books >= 0:
        # print(max((books_today + d - 1) % 7,1))
        if max((books_today + d - 1) % 7, 1)<6:
                have_books+=k
        books_today += 1
    else:
        books_today-=1
    # print('e',have_books, books_today)
with open('output.txt', 'w') as f2:
    f2.write(str(books_today))
# print(books_today)