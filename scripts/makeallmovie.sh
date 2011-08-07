#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things

# Steps:
#
# 1) Ensure followed createlinks.sh for each directory.
# 2) use the script

# order of models as to appear in final table
dircollect='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.liker2butbeta40 thickdiskrr2 run.like8 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 run.liker1 thickdiskr2 run.liker2 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdisk2 thickdisk3 runlocaldipole3dfiducial sasha99'

# note that thickdisk1 is actually bad, so ignore it.
# can choose so do runs in different order than collection.
#dirruns=$dircollect
# do expensive thickdisk7 and sasha99 last so can test things
dirruns='thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.liker2butbeta40 thickdiskrr2 run.like8 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 run.liker1 thickdiskr2 run.liker2 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdisk2 thickdisk3 runlocaldipole3dfiducial thickdisk7 sasha99'



#modelnamelist='A94BfN40 A94BfN40\_C1  A94BfN40\_C2  A94BfN40\_C3  A94BfN40\_C4  A94BfN40\_C5  A-94BfN10     A-94BfN10\_C1 A-5BfN10      A0BfN10       A5BfN10       A94BfN10      A94BfN10\_C1  A94BfN10\_R1  A-94BfN40     A94BpN10      A-94BtN10     A-5BtN10      A0BtN10       A5BtN10       A94BtN10      A94BtN10\_R1  MB09_D      '


EXPECTED_ARGS=7
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {moviedirname dolinks dofiles make1d make2d makemovie collect}"
    echo "e.g. sh makeallmovie.sh moviefinal1 1 1 1 1 1 1"
    exit $E_BADARGS
fi


# name of movie directory in each dirrun
moviedirname=$1
dolinks=$2
dofiles=$3
make1d=$4
make2d=$5
makemovie=$6
collect=$7


# On ki-jmck in /data2/jmckinne/
cd /data2/jmckinne/


###################################
if [ $dolinks -eq 1 ]
then

    echo "Doing Links"

for thedir in $dirruns
do
    
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
    cd /data2/jmckinne/${thedir}/$moviedirname/dumps/
    ln -s /data2/jmckinne/${thedir}/dumps/$firstfieldlinefile .
    ln -s /data2/jmckinne/${thedir}/dumps/$lastfieldlinefile .



done

    echo "Done with links"

fi


###################################
if [ $dofiles -eq 1 ]
then

    echo "Doing files"

for thedir in $dirruns
do

    echo "Doing files for: "$thedir


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
	nothing=1
        #in __init__.local.py:
        # for runlocaldipole3dfiducial myMB09=0 -> myMB09=1
	#sed 's/myMB09=0/myMB09=1/g' __init__.local.temp.py > __init__.local.py

	# MB09 no longer needs this with myMB09 switch
	#rm -rf titf.txt
	#echo "#" >> titf.txt
	#echo "0 1 2000 100000" >> titf.txt

	# switch no longer present or required as long as model name correct

    else
	cp __init__.local.temp.py __init__.local.py
    fi

    rm -rf __init__.local.temp.py


done

    echo "Done with files"


fi




##############################################
make1d2dormovie=$(( $make1d + $make2d + $makemovie ))
if [ $make1d2dormovie -gt 0 ]
then

    echo "Doing movie stuff"

    for thedir in $dirruns
    do
	echo "Doing makemovielocal for: "$thedir

	cd /data2/jmckinne/${thedir}/$moviedirname
	
        #run makemovie.sh 1 1 1 or similar
	sh makemovielocal.sh ${thedir} $make1d $make2d $makemovie
    done

    echo "Done with movie"

fi


##############################################
#
echo "Now collect Latex results"
if [ $collect -eq 1 ]
then

    cd /data2/jmckinne/

    echo "Doing collection"

    # refresh tables.tex
    rm  -rf tables$moviedirname.tex


    for thedir in $dircollect
    do
	echo "Doing collection for: "$thedir
	
	cat /data2/jmckinne/${thedir}/$moviedirname/python.plot.out | grep Latex >> tables$moviedirname.tex
    done


    ##############################################
    #
    # Normal tables:
    grep "Latex1:" tables$moviedirname.tex | sed 's/Latex1: //g' > table1$moviedirname.tex
    grep "Latex2:" tables$moviedirname.tex | sed 's/Latex2: //g' > table2$moviedirname.tex
    grep "Latex3:" tables$moviedirname.tex | sed 's/Latex3: //g' > table3$moviedirname.tex
    grep "Latex4:" tables$moviedirname.tex | sed 's/Latex4: //g' > table4$moviedirname.tex
    grep "Latex5:" tables$moviedirname.tex | sed 's/Latex5: //g' > table5$moviedirname.tex
    grep "Latex6:" tables$moviedirname.tex | sed 's/Latex6: //g' > table6$moviedirname.tex
    grep "Latex7:" tables$moviedirname.tex | sed 's/Latex7: //g' > table7$moviedirname.tex
    grep "Latex8:" tables$moviedirname.tex | sed 's/Latex8: //g' > table8$moviedirname.tex
    grep "Latex9:" tables$moviedirname.tex | sed 's/Latex9: //g' > table9$moviedirname.tex
    # Aux tables:
    grep "Latex95:" tables$moviedirname.tex | sed 's/Latex95: //g' > table95$moviedirname.tex
    grep "Latex96:" tables$moviedirname.tex | sed 's/Latex96: //g' > table96$moviedirname.tex
    grep "Latex97:" tables$moviedirname.tex | sed 's/Latex97: //g' > table97$moviedirname.tex
    grep "Latex99:" tables$moviedirname.tex | sed 's/Latex99: //g' > table99$moviedirname.tex

    echo "Done with collection"

fi

    



echo "Done with all stages"
