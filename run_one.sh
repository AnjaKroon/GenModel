U=1000
m=1000000
e=0.1
b=100
/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b} > time_${U}_${m}_${e}_${b}.txt
# somehow want to save the time from discrete.py
mv ModProbDist.png ModProbDist_${U}_${m}_${e}_${b}.png
mv Gen_Samples.npy Gen_Samples_${U}_${m}_${e}_${b}.npy