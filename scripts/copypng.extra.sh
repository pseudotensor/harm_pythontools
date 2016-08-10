#!/bin/bash

alias lssdir='ls -ap | grep / | sed "s/\///"'

list=`lssdir | grep jonharmrad`

oldpath=`pwd`

for mydir in $list
do
    echo $mydir
    cd $mydir/movienew2/
    scp fig3*.png fig2*.png jon@physics-179.umd.edu:/data/jon/pseudotensor@gmail.com/rad_papers/opacity/harm_dc/$mydir/
    cd $oldpath
done
