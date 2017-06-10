************************************************
This is for CSE 517 hw1 languange models
author : An Yan
date: Jan, 2017
Python version: 2.7
*************************************************

HOW TO RUN:

example:

python yanan_lm.py -t 1 brown.train.txt brown.dev.txt

****************************
parameters: 

-t: unk threshold, default = 1. 
test data set 
dev data set or test data set.

*****************************
Help message, please type


python yanan_lm.py -h

************************************************
note: 
if you run "python yanan_lm.py -t 1 brown.train.txt brown.dev.txt",
you will get perplexities on dev data only.
if you want to get result on test data. Go to modify the code.
Specifically, in the main function, comment out dev data piece, and 
uncomment the test data part. 

run "python yanan_lm.py -t 1 brown.train.txt brown.test.txt" 

****************************************************************
sample console output files included in this folder
