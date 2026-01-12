
def interpolate(l: list):
    i = 0
    new_l = []
    while i < len(l):
        new_l.append(l[i])
        if(i + 1 >= len(l)):
            i += 1
            continue
        middle_val = (l[i] + l[i+1]) / 2
        new_l.append(middle_val)
        i += 1

    return new_l

print(interpolate([1, 2, 3, 2, 1]))