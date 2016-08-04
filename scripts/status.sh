#!/bin/bash

alias lssdir='ls -ap | grep / | sed "s/\///"'

list=`lssdir | grep jonharmrad`

for mydir in $list
do
    echo $mydir
    file=`ls -alrt $mydir/images/|tail -2|head -1|awk '{print $9}'`

    head -2 $mydir/images/$file | tail -1
done
