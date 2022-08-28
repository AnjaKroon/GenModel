U=100
m=1000000
e=0.1
b=100
#/usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b}

for U in 100 1000 10000 100000 1000000
do
for m in  10 100 1000 10000 1000000 10000000
do
for b in 30 50 100 
do
echo /usr/local/bin/python3 /Users/anja/Documents/Flo_Research/GenModel/discrete.py ${U} ${m} ${e} ${b}
# when ready to do the big run remove echo
# mv ModProbDist.png ModProbDist_${U}_${m}_${e}_${b}.png # uncomment this for the big run
done # for b
done # for m
done # for U

# I want to make U: 100, 1000, 10000, 100000, 1000000 (5 options)
# I want to vary m:  10000000, 1000000, 10000, 1000, 100, 10 (6 options)
# We are going to keep e the same for all
# I want to vary b: 100, 50, 30 (3 options)
# 5 * 6 * 3 = 90 combinations

# What info do I want to pull? 
# I want to pull the modified plot, the run time, and the parameters inputted -- anything else?
