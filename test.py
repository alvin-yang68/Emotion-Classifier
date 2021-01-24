import msvcrt


counter = 1000000
while counter != 0:
    print(counter)
    if msvcrt.kbhit():
        break
    counter -= 1
