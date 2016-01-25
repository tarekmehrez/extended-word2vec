
count=0
mkdir figs-$1

for i in $(ls $1);
do
	echo $1/$i
	python main.py --plot 1 --model $1/$i --run $count
	count=$((count+1))
done

mv *.eps figs-$1
touch final-dist-$1.txt

for i in $(ls dist*)
do
	cat $i >> final-dist.txt
	echo '####################' >> final-dist.txt
done

mv final-dist-$1.txt figs-$1
rm dist*
