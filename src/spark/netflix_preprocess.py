from datetime import datetime
import pandas as pd
import numpy as np

import os
import random
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse


import random
if not os.path.isfile("NetflixRatings.csv"): 
#This line: "os.path.isfile("../Data/NetflixRatings.csv")" simply checks that is there a file with the name "NetflixRatings.csv" in the 
#in the folder "/Data/". If the file is present then it return true else false
    startTime = datetime.now()
    data = open("NetflixRatings_train.csv", mode = "w") #this line simply creates the file with the name "NetflixRatings.csv" in 
    data_test = open("NetflixRatings_test.csv", mode = "w")
    #write mode in the folder "Data".
    files = ['combined_data_1.txt','combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']
    #files = ['../Data/combined_data_2.txt', '../Data/combined_data_4.txt']
    for file in files:
        print("Reading from file: "+str(file)+"...")
        with open(file) as f:  #you can think of this command "with open(file) as f" as similar to 'if' statement or a sort of 
            #loop statement. This command says that as long as this file is opened, perform the underneath operation.
            for line in f:
                line = line.strip() #line.strip() clears all the leading and trailing spaces from the string, as here each line
                #that we are reading from a file is a string.
                #Note first line consist of a movie id followed by a semi-colon, then second line contains custID,rating,date
                #then third line agains contains custID,rating,date which belong to that movie ID and so on. The format of data
                #is exactly same as shown above with the heading "Example Data Point". Check out above.
                if line.endswith(":"):
                    movieID = line.replace(":", "") #this will remove the trailing semi-colon and return us the leading movie ID.
                else:
                    #here, in the below code we have first created an empty list with the name "row "so that we can insert movie ID 
                    #at the first position and rest customerID, rating and date in second position. After that we have separated all 
                    #four namely movieID, custID, rating and date with comma and converted a single string by joining them with comma.
                    #then finally written them to our output ".csv" file.
                    
                    row = [] 
                    row = [x for x in line.split(",")][:-1] #custID, rating and date are separated by comma
                    row.insert(0, movieID)
                    if random.randint(1, 10) >= 2:
                        data.write(",".join(row))
                        data.write("\n")
                    else:
                        data_test.write(",".join(row))
                        data_test.write("\n")
        print("Reading of file: "+str(file)+" is completed\n")
    data.close()
    print("Total time taken for execution of this code = "+str(datetime.now() - startTime))

