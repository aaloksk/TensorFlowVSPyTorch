#Kushum K C
#Date: Feb14 2023

#required packages
#install.packages('topsis')


#Step1: Setting working directory
path <- 'D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Project_3\\WD'
setwd(path)
getwd()

#Step2: Required Libraries
library(gdata) 
library(CGE)
library(topsis)


#Step3: Read CSV file
#Reads data from a CSV file where,
#the first row is the header of each column (criteria),
#the first column is the name of each row (alternatives), and
#the data is separated by a ","
data = read.csv("Result_OwnData.csv", header = TRUE, sep = ",", skip = 0, row.names = 1)
datmat = as.matrix(data) # puts the data into the form of a matrix

#Step4: Subjecting weights and positive negative benefits(impacts). 
#assigning the weights obtained from comparison matrix
wts = c(.739,0.21,0.51) #creates a numeric vector of the weights

#assign whether each of the criteria is a benefit (+) or negative benefit (-)
impcts = c('+','-','+') #creates a character vector


#Step5: Topsis analysis resulting the ranking based on weights
decision = topsis(datmat, wts, impcts)
decision   #to write the output ranking

#Decision by rank
dec_rank <- (decision[order(decision$rank),])
head(dec_rank)

#step 6: Extract the csv file of the results.
#write.csv(dec_rank, "rank_topsis.csv", row.names=FALSE)

















