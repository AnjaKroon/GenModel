U=1000
m=600
e=10 # decimals cause issues in the file name. Actual value is this divided by 100 and is compensated for in discrete.py and gen_S.py
b=100

/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b} > time_${U}_${m}_${e}_${b}.txt
/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/gen_S.py ${U} ${m} ${e} ${b} > S_stat_${U}_${m}_${e}_${b}.txt

mv ModProbDist.png ModProbDist_${U}_${m}_${e}_${b}.png 
mv Gen_Samples.npy Gen_Samples_${U}_${m}_${e}_${b}.npy 
mkdir Out_${U}_${m}_${e}_${b}
mv Gen_Samples_${U}_${m}_${e}_${b}.npy ModProbDist_${U}_${m}_${e}_${b}.png time_${U}_${m}_${e}_${b}.txt histo_${U}_${m}_${e}_${b}.csv S_stat_${U}_${m}_${e}_${b}.txt Out_${U}_${m}_${e}_${b}