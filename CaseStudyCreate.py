import csv
import pandas as pd
def calculate_average(dictionary):
    average_dict = {}
    for key, value in dictionary.items():
        # Calculate the average of the array of floats
        average = sum(value) / len(value)
        average_dict[key] = round(average,4)
    return average_dict

csv_file_path = "/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/RawData/(BEST)P3_Rico.csv"

hit1Dict = {}
hit5Dict = {}
hit10Dict = {}
mrrDict = {}

with open(csv_file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        #format: mrr, 1,5,10
        label = row[2]
        mrr = float(row[3])
        k1 = float(row[4])
        k5 = float(row[5])
        k10 = float(row[6])
        
        if label in mrrDict.keys():
            listAdd = mrrDict[label]
            listAdd.append(mrr)
            mrrDict[label] = listAdd
        if label not in mrrDict.keys():
            mrrDict[label] = [mrr]
        
        if label in hit1Dict.keys():
            listAdd = hit1Dict[label]
            listAdd.append(k1)
            hit1Dict[label] = listAdd
        if label not in hit1Dict.keys():
            hit1Dict[label] = [k1]
        
        if label in hit5Dict.keys():
            listAdd = hit5Dict[label]
            listAdd.append(k5)
            hit5Dict[label] = listAdd
        if label not in hit5Dict.keys():
            hit5Dict[label] = [k5]
        
        if label in hit10Dict.keys():
            listAdd = hit10Dict[label]
            listAdd.append(k10)
            hit10Dict[label] = listAdd
        if label not in hit10Dict.keys():
            hit10Dict[label] = [k10]
        
    
mrr_avg = calculate_average(mrrDict)
k1_avg = calculate_average(hit1Dict)
k5_avg = calculate_average(hit5Dict)
k10_avg = calculate_average(hit10Dict)

# Convert dictionaries to DataFrames
df1 = pd.DataFrame(list(mrr_avg.items()), columns=['Screen Type', 'MRR'])
df2 = pd.DataFrame(list(k1_avg.items()), columns=['Screen Type', 'Hits@1'])
df3 = pd.DataFrame(list(k5_avg.items()), columns=['Screen Type', 'Hits@5'])
df4 = pd.DataFrame(list(k10_avg.items()), columns=['Screen Type', 'Hits@10'])

# Concatenate DataFrames
df_concatenated = pd.concat([df1, df2['Hits@1'], df3['Hits@5'], df4['Hits@10']], axis=1)

# Write to CSV
df_concatenated.to_csv('/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/RawData/Case Study/VASE_Case_Rico.csv', index=False)
