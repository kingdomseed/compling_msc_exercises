
def filter_longer(in_file, out_file, max_len=8):
    print(f"filter_longer: {in_file}, {out_file}, {max_len}")

def count_lines(filename):
    print(f"count_lines: {filename}")

filter_longer("input.txt", "output.txt", max_len=10)
count_lines("input.txt")