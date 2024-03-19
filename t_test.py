# Importing library
import scipy.stats as stats
import csv

csv_source = "/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/RawData/S2V_RICO.csv"
csv_test   = "/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/RawData/(BEST)P3_Rico.csv"

source_mrr = []
source_k1 = []
source_k5 = []
source_k10 = []

with open(csv_source, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            information = row[3:]
            source_mrr.append(float(information[0]))
            source_k1.append(float(information[1]))
            source_k5.append(float(information[2]))
            source_k10.append(float(information[3]))
            
test_mrr = []
test_k1 = []
test_k5 = []
test_k10 = []

with open(csv_test, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            information = row[3:]
            test_mrr.append(float(information[0]))
            test_k1.append(float(information[1]))
            test_k5.append(float(information[2]))
            test_k10.append(float(information[3]))

mrr = stats.ttest_rel(source_mrr, test_mrr)
k1 = stats.ttest_rel(source_k1, test_k1)
k5 = stats.ttest_rel(source_k5, test_k5)
k10 = stats.ttest_rel(source_k10, test_k10)

print('PVal MRR: ', mrr[1])
print('PVal K1: ', k1[1])
print('PVal K5: ', k5[1])
print('PVal K10: ', k10[1])
