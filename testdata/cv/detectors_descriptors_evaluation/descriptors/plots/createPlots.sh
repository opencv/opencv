echo Creating plots for descriptors evaluation using gnuplot...

datasets=( bark bikes boat graf leuven trees ubc wall )

gpFile='plots.p'
cat > $gpFile <<End-of-message
set terminal png
set size 0.7, 0.7
set xlabel "1-precision"
set ylabel "recall"
set xr[0:1]
set yr[0:1]

set key box
set key left top
End-of-message


for dataset in ${datasets[@]}
do
	echo '' >> $gpFile
	echo set title \'$dataset\' >> $gpFile
	echo set output \'$dataset.png\' >> $gpFile
	echo -n 'plot ' >> $gpFile
	idx=0
	count=`ls *_$dataset.csv | wc -l`
	for file in `ls *_$dataset.csv` 
	do
		title=`echo $file | sed -e "s/_$dataset.csv//"`
		echo -n \'$file\' title \'$title\' with lines >> $gpFile
		idx=`expr $idx + 1`
		if [ $idx -ne $count ]
		then
			echo -n ', ' >> $gpFile
		else
			echo ''  >> $gpFile
		fi
	done
done

gnuplot $gpFile

echo Done.
