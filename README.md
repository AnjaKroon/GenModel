# GenModel
###### Evaluation Ground Truth project with Florence Regol

The program that currently does the desired computation is discrete.py. discrete.py takes a uniform probability distribution and allows the user to modify it and then generate samples from that modified probability distribution. 
It takes in the following parameters:
* U| represnting the probability space
* m the number of samples taken from the probability distribution
* e the error for this given application as defined in the overview pdf
* b the percentage of the origional probability 

discrete.py outputs the input parameters, the program run time, and a discrete probability distribution in a corresponding png. 
The objective of the shell scripts (any code ending in .sh) is to run discrete.py over many combinations of U, m and b. 

run_one.sh runs discrete.py with desired parameters
run_few.sh runs discrete.py with 8 combinations of input parameters
run_many.sh runs discrete.py with 90 combinations of input parameters

In run_few.sh, I was able to successfully pipe all input paramters and run time into time.txt. I was also able to successfully make corresponding .png files called 'ModProbDist_U_m_e_b.png' in which the input parameters in the file name are replaced by the numeric values of its test case. 
