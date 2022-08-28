# GenModel
###### Evaluation Ground Truth project with Florence Regol

## Objective
discrete.py takes a uniform probability distribution and allows the user to modify it and then generate samples from that modified probability distribution. 
It takes in the following parameters:
* |U| represnting the probability space
* m the number of samples taken from the probability distribution
* e the error for this given application as defined in the overview pdf
* b the percentage of the origional probability 

discrete.py outputs the input parameters, the program run time, and a discrete probability distribution in a corresponding png. 
The objective of the shell scripts (any code ending in .sh) is to run discrete.py over many combinations of U, m and b. 

run_one.sh runs discrete.py with desired parameters
run_few.sh runs discrete.py with 8 combinations of input parameters
run_many.sh runs discrete.py with 90 combinations of input parameters

In run_few.sh, I was able to successfully pipe all input paramters and run time into time.txt. I was also able to successfully make corresponding .png files called 'ModProbDist_U_m_e_b.png' in which the input parameters in the file name are replaced by the numeric values of its test case. 

## Limitations
Currently, this code has some limitations to robustness that I will describe here. 
* As input, discrete.py can only work with a uniform probability distribution. 
* A .png file is currently created representing the modified probability distribution. Creation of this png increases the processing time. If no such images are needed, it would be advantageous to remove this capability for running the program over large combinations of input parameters. 

## What are the other files for?
* Evaluation_ground_truth_FloRegol.pdf -- The origional description of the research problem. Describes how this code fits into a larger research problem. 
* ExampleofModProbPlot.png -- The output .png file with input parameters of U = 1000, m = 1000, e = 0.1, and b = 100. 
* continuous.py -- Practice with continuous probability distributions, sampling them, and beginning to mix them together. Not relevant here anymore. Should probably remove!
* rand_continuous.py -- Same as continuous.py but with random gen seed
* stat_test.py -- The origional starter code linked to the origional pdf document
* stat_test_AK.py -- Some modifications to the origional stat_test.py code that helped me understand what that origional code did.
