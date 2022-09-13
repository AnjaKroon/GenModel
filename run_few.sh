e=10

for U in 100 1000 # iterates over the two listed values for U
do
case ${U} in # we are only interested in the case where U is more than m!

  100)
    MLOOP="5 10 50" # define a string for values in second loop
    ;;

  1000)
    MLOOP="10 100 500"
    ;;

  *)
    echo "unknown value of U ${U}"
    exit
    ;;
esac # case written backwards is the end
for m in  ${MLOOP}
do
for b in 50 100 
do
echo "python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b}"
python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b}
echo "python3 /Users/anja/Documents/Flo_Research/GenModel/gen_S.py ${U} ${m} ${e} ${b} > S_stat_${U}_${m}_${e}_${b}.txt" # so user knows what's happening while waiting
python3 /Users/anja/Documents/Flo_Research/GenModel/gen_S.py ${U} ${m} ${e} ${b} > S_stat_${U}_${m}_${e}_${b}.txt 

# mv ModProbDist.png ModProbDist_${U}_${m}_${e}_${b}.png # uncomment this for the big run
mv Gen_Samples.npy Gen_Samples_${U}_${m}_${e}_${b}.npy
cat S_stat_*.txt >> S_stats.txt # I want to label this with the date and time
mkdir Out_${U}_${m}_${e}_${b}
mv Gen_Samples_${U}_${m}_${e}_${b}.npy \
histo_${U}_${m}_${e}_${b}.csv \
S_stat_${U}_${m}_${e}_${b}.txt \
Out_${U}_${m}_${e}_${b}/ #ModProbDist_${U}_${m}_${e}_${b}.png can decide to add back in but adds a lot of time

done # for b
done # for m
done # for U


