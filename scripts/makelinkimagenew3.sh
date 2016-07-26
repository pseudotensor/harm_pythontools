list=`ls lrhosmall*.png`

ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagesmall$ii.png
done

export fps=25

rm -rf imagesmall.mp4
#avconv -i imagesmall%d.png -r 25 imagesmall.mp4
ffmpeg -y -fflags +genpts -r $fps -i imagesmall%d.png -vcodec mpeg4 imagesmall.$modelname.mp4


list=`ls lrhovsmall*.png`

ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagevsmall$ii.png
done

rm -rf imagevsmall.mp4
#avconv -i imagevsmall%d.png -r 25 imagevsmall.mp4
ffmpeg -y -fflags +genpts -r $fps -i imagevsmall%d.png -vcodec mpeg4 imagevsmall.$modelname.mp4

#exit

list=`ls lrho*.png | grep -v lrhosmall | grep -v lrhovsmall | grep -v lrhobig`

ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagenormal$ii.png
done

rm -rf imagenormal.mp4
#avconv -i imagenormal%d.png -r 25 imagenormal.mp4
ffmpeg -y -fflags +genpts -r $fps -i imagenormal%d.png -vcodec mpeg4 imagenormal.$modelname.mp4


list=`ls lrhobig*.png`

ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagebig$ii.png
done

rm -rf imagebig.mp4
avconv -i imagebig%d.png -r 25 imagebig.mp4
ffmpeg -y -fflags +genpts -r $fps -i imagebig%d.png -vcodec mpeg4 imagebig.$modelname.mp4


