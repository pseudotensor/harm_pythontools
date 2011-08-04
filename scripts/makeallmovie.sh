#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things

# note that thickdisk1 is actually bad, so ignore it.
dirruns='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.liker2butbeta40 thickdiskrr2 run.like8 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 run.liker1 thickdiskr2 run.liker2 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdisk2 thickdisk3 runlocaldipole3dfiducial'


EXPECTED_ARGS=5
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {moviedirname dolinks make1d make2d makemovie}"
    echo "e.g. sh makeallmovie.sh moviefinal1 1 1 1 1"
    exit $E_BADARGS
fi


# name of movie directory in each dirrun
moviedirname=$1
dolinks=$2
make1d=$3
make2d=$4
makemovie=$5



# On ki-jmck in /data2/jmckinne/
cd /data2/jmckinne/

# 1) Ensure followed createlinks.sh for each directory.

for thedir in $dirruns
do
    echo $thedir

    if [ $dolinks -eq 1 ]
    then

    echo "Doing links for: "$thedir


    # make movie directory
    mkdir -p /data2/jmckinne/${thedir}/$moviedirname/dumps/
    cd /data2/jmckinne/${thedir}/$moviedirname

    echo "clean movie directory in case already existed: "$thedir
    rm -rf /data2/jmckinne/${thedir}/$moviedirname/*.png /data2/jmckinne/${thedir}/$moviedirname/*.eps /data2/jmckinne/${thedir}/$moviedirname/*.npy /data2/jmckinne/${thedir}/$moviedirname/python*.out /data2/jmckinne/${thedir}/$moviedirname/*.avi /data2/jmckinne/${thedir}/$moviedirname/*.pdf

    echo "create new links for: "$thedir
    alias cp='cp'
    cp ~/py/scripts/createlinksalt.sh .
    sh createlinksalt.sh 1 /data2/jmckinne/${thedir} ./

    echo "now remove some fraction of links (only keep about 50 for averaging period, first one, and last one): "$thedir
    cd /data2/jmckinne/${thedir}/$moviedirname/dumps/
    # avoid @ symbol on soft links
    alias ls='ls'
    # get list of files in natural human order
    fieldlinelist=`ls -v | grep "fieldline"`
    firstfieldlinefile=`ls -v | grep "fieldline" | head -1`
    lastfieldlinefile=`ls -v | grep "fieldline" | tail -1`
    numfiles=`echo $fieldlinelist | wc | awk '{print $2}'`
    # set 1/2 to keep since average over roughly latter half in time of data
    keepfilesstart=$(( (1 * $numfiles) / 2 ))
    keepfilesend=$(( $numfiles ))
    keepfilesdiff=$(( $keepfilesend - $keepfilesstart ))
    #keepfieldlinelist=`ls -v | grep "fieldline" | tail -$keepfilesstart | head -$keepfilesdiff`
    rmfieldlinelist=`ls -v | grep "fieldline" | head -$keepfilesstart | tail -$keepfilesdiff`
    for fil in $rmfieldlinelist
    do
	rm -rf /data2/jmckinne/${thedir}/$moviedirname/dumps/$fil
    done
    #
    echo "now trim every so a file so only 50+2 files in the end: "$thedir
    fieldlinelist=`ls -v | grep "fieldline"`
    numfiles=`echo $fieldlinelist | wc | awk '{print $2}'`
    numkeep=52
    skipfactor=$(( $numfiles / $numkeep ))
    iiter=0
    for fil in $fieldlinelist
    do
	mymod=$(( $iiter % $skipfactor ))
	if [ $mymod -ne 0 ]
	    then
	    rm -rf /data2/jmckinne/${thedir}/$moviedirname/dumps/$fil
	fi
	iiter=$(( $iiter + 1 ))
    done
    #
    echo "Ensure fieldline0000.bin and last fieldline files exist: "$thedir
    ln -s /data2/jmckinne/${thedir}/$firstfieldlinefile .
    ln -s /data2/jmckinne/${thedir}/$lastfieldlinefile .



    echo "cp makemovie.sh: "$thedir
    cd /data2/jmckinne/${thedir}/$moviedirname/
    cp ~/py/scripts/makemovie.sh .

    echo "edit makemovie.sh: "$thedir
    #in makemovie.sh:
    # for thickdisk7 runn=5
    # for run.like8 run.liker1 run.liker2 runn=12
    # for runlocaldipole3dfiducial runn=12
    if [ "$thedir" == "thickdisk7" ]
	then
	sed -e 's/export runn=[0-9]*/export runn=5/g' makemovie.sh > makemovielocal.temp.sh
    else
	sed -e 's/export runn=[0-9]*/export runn=12/g' makemovie.sh > makemovielocal.temp.sh
    fi

    # force use of local __init__.py file:
    sed 's/export initfile=\$MREADPATH\/__init__.py/export initfile=\/data2\/jmckinne\/'${thedir}'\/'${moviedirname}'\/__init__.local.py/g' makemovielocal.temp.sh > makemovielocal.sh
    rm -rf makemovielocal.temp.sh

    echo "cp  __init__.py to __init__.local.py: "$thedir
    cp ~/py/mread/__init__.py __init__.local.temp.py

    if [ "$thedir" == "runlocaldipole3dfiducial" ]
    then
        #in __init__.local.py:
        # for runlocaldipole3dfiducial myMB09=0 -> myMB09=1
	sed 's/myMB09=0/myMB09=1/g' __init__.local.temp.py > __init__.local.py
    else
	cp __init__.local.temp.py __init__.local.py
    fi

    rm -rf __init__.local.temp.py




    fi



    echo "Doing makemovielocal for: "$thedir

    cd /data2/jmckinne/${thedir}/$moviedirname

    #run makemovie.sh 1 1 1 or similar
    sh makemovielocal.sh $make1d $make2d $makemovie


done


echo "Done with all rundirs"


