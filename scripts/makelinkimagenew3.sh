#!/bin/bash

modelname=$1

list=`ls lrhosmall*.png`

rm -rf imagesmall*.png
ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagesmall$ii.png
done

export fps=25

rm -rf imagesmall$modelname.mp4
#avconv -i imagesmall%d.png -r 25 imagesmall$modelname.mp4
ffmpeg -y -fflags +genpts -r $fps -i imagesmall%d.png -vcodec mpeg4 imagesmall.$modelname.mp4
rm -rf imagesmall*.png


list=`ls lrhovsmall*.png`

rm -rf imagevsmall*.png
ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagevsmall$ii.png
done

rm -rf imagevsmall$modelname.mp4
#avconv -i imagevsmall%d.png -r 25 imagevsmall$modelname.mp4
ffmpeg -y -fflags +genpts -r $fps -i imagevsmall%d.png -vcodec mpeg4 imagevsmall.$modelname.mp4
rm -rf imagevsmall*.png

list=`ls lrho*.png | grep -v lrhosmall | grep -v lrhovsmall | grep -v lrhobig`

rm -rf imagenormal*.png
ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagenormal$ii.png
done

rm -rf imagenormal$modelname.mp4
#avconv -i imagenormal%d.png -r 25 imagenormal$modelname.mp4
ffmpeg -y -fflags +genpts -r $fps -i imagenormal%d.png -vcodec mpeg4 imagenormal.$modelname.mp4
rm -rf imagenormal*.png


list=`ls lrhobig*.png`

rm -rf imagebig*.png
ii=0
for fil in $list
do
    echo $fil
    ii=$(($ii+1))
    ln -s $fil imagebig$ii.png
done

rm -rf imagebig$modelname.mp4
#avconv -i imagebig%d.png -r 25 imagebig$modelname.mp4
ffmpeg -y -fflags +genpts -r $fps -i imagebig%d.png -vcodec mpeg4 imagebig.$modelname.mp4
rm -rf imagebig*.png


