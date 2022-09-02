U=100
m=10
e=10 # decimals cause issues in the file name. Actual value is this divided by 100 so multiply e by 100
b=100
/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b} > time_${U}_${m}_${e}_${b}.txt
# need to call gen_S here 
/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/gen_S.py ${U} ${m} ${e} ${b}
# somehow want to save the time from discrete.py
mv ModProbDist.png ModProbDist_${U}_${m}_${e}_${b}.png
mv Gen_Samples.npy Gen_Samples_${U}_${m}_${e}_${b}.npy 