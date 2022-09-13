U=100


python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b} 
echo "python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b}"

python3 /Users/anja/Documents/Flo_Research/GenModel/gen_S.py ${U} ${m} ${e} ${b} > S_stat_${U}_${m}_${e}_${b}.txt
echo "python3 /Users/anja/Documents/Flo_Research/GenModel/gen_S.py ${U} ${m} ${e} ${b} > S_stat_${U}_${m}_${e}_${b}.txt"

python3 /Users/anja/Documents/Flo_Research/GenModel/flo_poisson.py ${U} ${m} ${e} ${b} ${t} 
echo "python3 /Users/anja/Documents/Flo_Research/GenModel/flo_poisson.py ${U} ${m} ${e} ${b} ${t}"

#mv ModProbDist.png ModProbDist_${U}_${m}_${e}_${b}.png 
mv Gen_Samples.npy Gen_Samples_${U}_${m}_${e}_${b}.npy 
mkdir Out_${U}_${m}_${e}_${b}
mv Gen_Samples_${U}_${m}_${e}_${b}.npy histo_${U}_${m}_${e}_${b}.csv S_stat_${U}_${m}_${e}_${b}.txt Out_${U}_${m}_${e}_${b}