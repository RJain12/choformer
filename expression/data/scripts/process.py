import csv

g2e_file = "rna/cleaned GSE77800/final_sum_nozero_log.csv"
main_file = "train.csv"

g2e = dict()

with open(g2e_file, "r") as f:
    reader = csv.reader(f)
    for ix, i in enumerate(reader):
        if ix == 0: continue
        g2e[int(i[0])] = i[1]

main_data = []

with open(main_file, "r") as f:
    reader = csv.reader(f)
    for ix, i in enumerate(reader):
        try:
            if ix == 0: continue
            main_data.append([i[2], g2e[int(i[0])]])
        except: continue

with open("test_cleaned.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Sequence", "Expression", "ID", "FID"])
    for i in main_data:
        writer.writerow(i)