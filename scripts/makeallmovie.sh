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
#dirruns='sasha99'



#modelnamelist='A94BfN40 A94BfN40\_C1  A94BfN40\_C2  A94BfN40\_C3  A94BfN40\_C4  A94BfN40\_C5  A-94BfN10     A-94BfN10\_C1 A-5BfN10      A0BfN10       A5BfN10       A94BfN10      A94BfN10\_C1  A94BfN10\_R1  A-94BfN40     A94BpN10      A-94BtN10     A-5BtN10      A0BtN10       A5BtN10       A94BtN10      A94BtN10\_R1  MB09_D      '


EXPECTED_ARGS=9
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {moviedirname dolinks dofiles make1d makemerge makeplot makeframes makemovie collect}"
    echo "e.g. sh makeallmovie.sh moviefinal1 1 1 1 1 1 1 1 1"
    exit $E_BADARGS
fi


# name of movie directory in each dirrun
moviedirname=$1
dolinks=$2
dofiles=$3
make1d=$4
makemerge=$5
makeplot=$6
makeframes=$7
makemovie=$8
collect=$9


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

    # number of files to keep
    numkeep=150

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
    # if above 1/2 kills more than want to keep, then avoid kill of 1/2
    if [ $keepfilesdiff -lt $numkeep ]
	then
	keepfilesstart=0
	keepfilesend=$numfiles
	keepfilesdiff=$(( $keepfilesend - $keepfilesstart ))
    else
        #keepfieldlinelist=`ls -v | grep "fieldline" | tail -$keepfilesstart | head -$keepfilesdiff`
	rmfieldlinelist=`ls -v | grep "fieldline" | head -$keepfilesstart | tail -$keepfilesdiff`
	for fil in $rmfieldlinelist
	do
	    rm -rf /data2/jmckinne/${thedir}/$moviedirname/dumps/$fil
	done
    fi
    #
    echo "now trim every so a file so only about numkeep+2 files in the end: "$thedir
    fieldlinelist=`ls -v | grep "fieldline"`
    numfiles=`echo $fieldlinelist | wc | awk '{print $2}'`
    #
    skipfactor=$(( $numfiles / $numkeep ))
    if [ $skipfactor -eq 0 ]
    then
	resid=$(( $numfiles - $numkeep ))
	echo "keeping bit extra: "$resid
    fi
    if [ $skipfactor -gt 0 ]
    then
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
    fi
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
    elif [ "$thedir" == "sasha99" ]
	then
	    sed -e 's/export runn=[0-9]*/export runn=8/g' makemovie.sh > makemovielocal.temp.sh
    else
	    sed -e 's/export runn=[0-9]*/export runn=12/g' makemovie.sh > makemovielocal.temp.sh
    fi

    # force use of local __init__.py file:
    sed 's/export initfile=\$MREADPATH\/__init__.py/export initfile=\/data2\/jmckinne\/'${thedir}'\/'${moviedirname}'\/__init__.local.py/g' makemovielocal.temp.sh > makemovielocal.sh
    rm -rf makemovielocal.temp.sh

    echo "cp  __init__.py to __init__.local.py: "$thedir
    cp ~/py/mread/__init__.py __init__.local.temp.py

    #if [ "$thedir" == "runlocaldipole3dfiducial" ]
    #then
	#nothing=1
        #in __init__.local.py:
        # for runlocaldipole3dfiducial myMB09=0 -> myMB09=1
	#sed 's/myMB09=0/myMB09=1/g' __init__.local.temp.py > __init__.local.py

	# MB09 no longer needs this with myMB09 switch
	#rm -rf titf.txt
	#echo "#" >> titf.txt
	#echo "0 1 2000 100000" >> titf.txt

	# switch no longer present or required as long as model name correct

    #else
	#    cp __init__.local.temp.py __init__.local.py
    #fi

	cp __init__.local.temp.py __init__.local.py

    rm -rf __init__.local.temp.py


done

    echo "Done with files"


fi




##############################################
make1d2dormovie=$(( $make1d + $makemerge + $makeplot + $makeframes + $makemovie ))
if [ $make1d2dormovie -gt 0 ]
then

    echo "Doing makemovie.sh stuff"

    for thedir in $dirruns
    do
	echo "Doing makemovielocal for: "$thedir

	cd /data2/jmckinne/${thedir}/$moviedirname
	
	sh makemovielocal.sh ${thedir} $make1d $makemerge $makeplot $makeframes $makemovie
    done

    echo "Done with makemovie.sh stuff"

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

    iiter=1
    for thedir in $dircollect
    do
	    echo "Doing collection for: "$thedir
        if [ $iiter -eq 1 ]
        then
		    cat /data2/jmckinne/${thedir}/$moviedirname/python.plot.out | grep "HLatex" >> tables$moviedirname.tex
        fi
		cat /data2/jmckinne/${thedir}/$moviedirname/python.plot.out | grep "VLatex" >> tables$moviedirname.tex

        iiter=$(( $iiter+1))
    done


    ##############################################
    #
    # Normal tables:
    grep "Latex1:" tables$moviedirname.tex | sed 's/[HV]Latex1: //g'  | column  -t > table1$moviedirname.tex
    grep "Latex2:" tables$moviedirname.tex | sed 's/[HV]Latex2: //g'  | column  -t > table2$moviedirname.tex
    grep "Latex3:" tables$moviedirname.tex | sed 's/[HV]Latex3: //g'  | column  -t > table3$moviedirname.tex
    grep "Latex4:" tables$moviedirname.tex | sed 's/[HV]Latex4: //g'  | column  -t > table4$moviedirname.tex
    grep "Latex5:" tables$moviedirname.tex | sed 's/[HV]Latex5: //g'  | column  -t > table5$moviedirname.tex
    grep "Latex6:" tables$moviedirname.tex | sed 's/[HV]Latex6: //g'  | column  -t > table6$moviedirname.tex
    grep "Latex7:" tables$moviedirname.tex | sed 's/[HV]Latex7: //g'  | column  -t > table7$moviedirname.tex
    grep "Latex8:" tables$moviedirname.tex | sed 's/[HV]Latex8: //g'  | column  -t > table8$moviedirname.tex
    grep "Latex9:" tables$moviedirname.tex | sed 's/[HV]Latex9: //g'  | column  -t > table9$moviedirname.tex
    # Aux tables:
    grep "Latex95:" tables$moviedirname.tex | sed 's/[HV]Latex95: //g'  | column  -t > table95$moviedirname.tex
    grep "Latex96:" tables$moviedirname.tex | sed 's/[HV]Latex96: //g'  | column  -t > table96$moviedirname.tex
    grep "Latex97:" tables$moviedirname.tex | sed 's/[HV]Latex97: //g'  | column  -t > table97$moviedirname.tex
    grep "Latex99:" tables$moviedirname.tex | sed 's/[HV]Latex99: //g'  | column  -t > table99$moviedirname.tex

    echo "Done with collection"

fi

    



echo "Done with all stages"
