
count=0
mkdir figs-$1

for i in $(ls $1);
do
	echo $1/$i
	python main.py --plot save --model $1/$i --run $count
	count=$((count+1))
done

mv *.eps figs-$1
touch final-dist-$1.txt

for i in $(ls dist*)
do
	cat $i >> final-dist-$1.txt
	echo '####################' >> final-dist-$1.txt
done

mv final-dist-$1.txt figs-$1
rm dist*
