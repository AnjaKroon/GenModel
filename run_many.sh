U=1000000
#m=1000000
e=0.1
b=100
#/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b}
# rm -f time.txt #-f means force it so if not there then no error message

for m in 1 10 100 1000 10000 1000000 10000000
do
/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b}
/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/gen_S.py ${U} ${m} ${e} ${b} > S_stat_${U}_${m}_${e}_${b}.txt

# mv ModProbDist.png ModProbDist_${U}_${m}_${e}_${b}.png # uncomment this for the big run, if add back in add to longer mv command later on
mv Gen_Samples.npy Gen_Samples_${U}_${m}_${e}_${b}.npy
#cat time_*.txt >> time.txt
cat S_stat_*.txt >> S_stats.txt
mkdir Out_${U}_${m}_${e}_${b}
mv Gen_Samples_${U}_${m}_${e}_${b}.npy histo_${U}_${m}_${e}_${b}.csv Out_${U}_${m}_${e}_${b}/

done # for m
