#!/bin/bash

alias lssdir='ls -ap | grep / | sed "s/\///"'

list=`lssdir | grep jonharmrad`

oldpath=`pwd`

for mydir in $list
do
    echo $mydir
    cd $mydir/movienew2/
    lastinit=`ls -rt |grep __init__|tail -1`
    cp ~/py/mread/__init__.py $lastinit
    cd $oldpath
done
