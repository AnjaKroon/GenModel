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

gen_S.py takes as input the generated samples from discrete.py as a numpy file. It then calculates the corresponding s-statisitc and creates a histogram with a dictionary datastructure representing each value in the probability space and the frequency in which it appears in the generated samples. The histogram output of this code is a csv file in the format of [[ item in prob space, frequency in samples array], ... ]. The s-statisitc is fed into a .txt file with the input parameters and is recorded for comparison later.

flo_poisson.py removes the dependency between generated samples via poissonization and returns
the s-statistic when all samples are independent. The inputs are U, m, ε, and b as described above. The output was previously just the s-statistic when 'poissonization' has occured. Recent changes have taken place to now output graphs running the experiment for multiple trials and plotting the resultant s statistic against the probability space |U|. 

run_one.sh runs discrete.py with desired parameters
run_few.sh runs discrete.py with a small number (~10) combinations of input parameters
run_many.sh runs discrete.py with a large number (~100) combinations of input parameters

## Code Output
In run_one.sh, the following output files are made:
* Gen_Samples_U_m_(e*100)_b.npy
* ModProbDist__U_m_(e*100)_b.png
* histo_U_m_(e*100)_b.csv

All .sh files move these four files into a folder called Out_U_m_e_b for easy tracking of files. 

In run_few and run_many, two summary files are made:
S_stats.txt describes the test run with the parameters U, m, e and b and the s statistic computed with the empirical probability.

This repo has an example of this structure in a folder called "Ran_Few_Output_Files" which include the results after run_few.sh was run. 

## Limitations
Currently, this code has some limitations to robustness that I will describe here. 
* As input, discrete.py can only work with a uniform probability distribution. 
* More robust testing of code could be useful to address potential unseen edge cases

## Next Steps
At this point, the problem statement has been defined and the foundational code base has been laid. The next step would be the first iteration of testing, calculating the s-statistic over numerous input parameters to see trends in the behavior.


## What are the other files for?
* Evaluation_ground_truth_FloRegol.pdf -- The origional description of the research problem. Describes how this code fits into a larger research problem. 
* ExampleofModProbPlot.png -- The output .png file with input parameters of U = 1000, m = 1000, e = 0.1, and b = 100. 
* continuous.py -- Practice with continuous probability distributions, sampling them, and beginning to mix them together. Not relevant here anymore. Should probably remove!
* rand_continuous.py -- Same as continuous.py but with random gen seed
* stat_test.py -- The origional starter code linked to the origional pdf document
* stat_test_AK.py -- Some modifications to the origional stat_test.py code that helped me understand what that origional code did.
