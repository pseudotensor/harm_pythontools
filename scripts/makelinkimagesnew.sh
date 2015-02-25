list=`ls lrhosmall*.png`

ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagesmall$ii.png
done

rm -rf imagesmall.mp4
avconv -i imagesmall%d.png -r 25 imagesmall.mp4

#exit

list=`ls lrho*.png | grep -v lrhosmall`

ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagenormal$ii.png
done

rm -rf imagenormal.mp4
avconv -i imagenormal%d.png -r 25 imagenormal.mp4

