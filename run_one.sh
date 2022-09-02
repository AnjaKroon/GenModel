U=100
m=10
e=10 # decimals cause issues in the file name. Actual value is this divided by 100 and is compensated for in discrete.py and gen_S.py
b=100

/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b} > time_${U}_${m}_${e}_${b}.txt
/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/gen_S.py ${U} ${m} ${e} ${b} > histo_time_${U}_${m}_${e}_${b}.csv

mv ModProbDist.png ModProbDist_${U}_${m}_${e}_${b}.png
mv Gen_Samples.npy Gen_Samples_${U}_${m}_${e}_${b}.npy 
