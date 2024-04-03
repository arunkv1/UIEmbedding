import csv
from UI_Embedding_main import *
import time
start_time = time.time()
testEmbeds = []
trainEmbeds = []
csv_file_path = ""
totalCtr = 0


ErrorCounter = 0

with open("rips_avgustdata.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["File Path", "Directory Name", "Testing", "embedding"])
    
    with open(csv_file_path, mode='r', newline='') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        rowCount = 0
        # Loop through each row in the CSV file
        for row in csv_reader:
            if rowCount > 0:
                # Check if the row has at least two columns (file path and label)
                
                if len(row) >= 0:
                    # Extract the file path and label
                    file_path = row[0]
                    label = row[2]
                    train = row[3]
                    embedding = makeEmbedding(file_path, 'regular') 
                    row = [file_path, label, train, embedding]
                    print("Write embedding Len: " , len(embedding))
                    row.append(embedding)
                    if train == "test":
                        csv_writer.writerow(row)
                        #testEmbeds.append(row)
                    elif train == "train":
                        csv_writer.writerow(row)
                        #trainEmbeds.append(row)
                    embedding = []

            rowCount += 1
            if totalCtr % 100 == 0:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " + str(totalCtr))
            totalCtr+=1
       
        print(totalCtr)

print(len(testEmbeds))
print(len(trainEmbeds))
print("Error Counter: ", ErrorCounter)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Time Elapsed: {elapsed_time} seconds")






