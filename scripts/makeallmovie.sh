#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things

# note that ubuntu defaults to dash after update.  Also causes \b to appear as ^H unlike bash.  Can't do \\b either -- still appears as ^H

# Steps:
#
# 1) Ensure followed createlinks.sh for each directory.
# 2) use the script

# order of models as to appear in final table
# SKIP thickdisk15 thickdisk2 since not run long enough

# POLOIDAL:
#dircollect='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'
# TOROIDAL:
#dircollecttoroidal='thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3'

# ALL:
dircollect='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'


# note that thickdisk1 is actually bad, so ignore it.
# can choose so do runs in different order than collection.
#dirruns=$dircollect
# do expensive thickdisk7 and sasha99 last so can test things
dirruns='thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3 thickdisk17 thickdisk10 thickdiskr15 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9full2pi sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'



#dirruns='thickdisk8'

#dirruns='thickdisk15 thickdiskr15 thickdisk2 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

#dirruns='thickdiskhr3'

#dirruns='thickdisk8'

#dirruns='sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

#dirruns='thickdisk7 sasha99'
#dirruns='runlocaldipole3dfiducial'
#dirruns='sasha99'

# number of files to keep
numkeep=350


EXPECTED_ARGS=20
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {moviedirname docleanexist dolinks dofiles make1d makemerge makeplot makemontage makepowervsmplots makespacetimeplots makefftplot makespecplot makeinitfinalplot makethradfinalplot makeframes makemovie makeavg makeavgmerge makeavgplot collect}"
    echo "e.g. sh makeallmovie.sh moviefinal1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
    exit $E_BADARGS
fi



# /lustre/ki/orange/jmckinne/thickdisk7/movie6
# sh makemovie.sh thickdisk7 1 1 1 0 1 1 0 0 0 0 0 
# jobstokill=`bjobs -u jmckinne -q kipac-ibq | awk '{print $1}'`
# for fil in $jobstokill ; do bkill -r $fil ; done 

# 
# /u/ki/jmckinne/nfsslac2/thickdisk7/movie6b
# sh makemovie.sh thickdisk7 1 1 1 0 1 1 0 0 0 0 0 
# jobstokill=`bjobs -u jmckinne -q kipac-gpuq | awk '{print $1}'`
# for fil in $jobstokill ; do bkill -r $fil ; done 

# name of movie directory in each dirrun
moviedirname=$1
docleanexist=$2
dolinks=$3
dofiles=$4
make1d=$5
makemerge=$6
makeplot=$7
makemontage=$8
makepowervsmplots=$9
makespacetimeplots=${10}
makefftplot=${11}
makespecplot=${12}
makeinitfinalplot=${13}
makethradfinalplot=${14}
makeframes=${15}
makemovie=${16}
makeavg=${17}
makeavgmerge=${18}
makeavgplot=${19}
collect=${20}


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

    echo "create new links (old links are removed) for: "$thedir


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
    #
    # set 1/2 to keep since average over roughly latter half in time of data
    #
    #default
    # most steady by 8000 but run till only 13000 for dipoley runs
    # or for toroidal runs, ran for 2X when was steady.  So also good.
    useend=1
    usefirst=1
    factor=2
    #
    if [ "$thedir" == "thickdisk7" ]
    then
        factor=3
    fi
    if [ "$thedir" == "sasha99" ]
    then
        factor=4
    fi
    if [ "$thedir" == "sasham9full2pi" ]
    then
        factor=3
    fi
    if [ "$thedir" == "sasham5" ]
    then
        factor=3
    fi
    if [ "$thedir" == "sasha0" ]
    then
        factor=3
    fi
    if [ "$thedir" == "sasha1" ]
    then
        factor=3
    fi
    if [ "$thedir" == "sasha9b25" ]
    then
        factor=3
    fi
    if [ "$thedir" == "sasha9b100" ]
    then
        factor=3
    fi
    if [ "$thedir" == "sasha9b200" ]
    then
        factor=4
    fi
    if [ "$thedir" == "thickdisk17" ]
    then
        factor=3
    fi
    if [ "$thedir" == "thickdisk3" ]
    then
        factor=3
    fi
    if [ "$thedir" == "thickdiskhr3" ]
    then
        factor=100000000
    fi
    #
    keepfilesstart=$(( (1 * $numfiles) / $factor ))
    keepfilesend=$(( $numfiles ))
    #
    # don't want to go till very end with this model
    if [ "$thedir" == "runlocaldipole3dfiducial" ]
    then
        useend=1
        keepfilesstart=$(( (900/5662)*$numfiles ))
        keepfilesend=$(( (2500/5662)*$numfiles ))
    fi
    #
    #
    #
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
    if [ $usefirst -eq 1 ]
    then
        ln -s /data2/jmckinne/${thedir}/dumps/$firstfieldlinefile .
    fi
    if [ $useend -eq 1 ]
    then
        ln -s /data2/jmckinne/${thedir}/dumps/$lastfieldlinefile .
    fi



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
    # more like runn=4 for thickdisk7 for avg creation.
    if [ "$thedir" == "thickdisk7" ]
	then
	    sed -e 's/export runn=[0-9]*/export runn=4/g' makemovie.sh > makemovielocal.temp.sh
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


    if [ "$thedir" == "thickdiskhr3" ]
	then
        ln -s /data2/jmckinne/thickdisk3/$moviedirname/qty2.npy /data2/jmckinne/${thedir}/$moviedirname/qty2_thickdisk3.npy
    fi


done

    echo "Done with files"


fi




##############################################
make1d2dormovie=$(( $make1d + $makemerge + $makeplot + $makemontage + $makepowervsmplots + $makespacetimeplots + $makefftplot + $makespecplot + $makeinitfinalplot + $makethradfinalplot + $makeframes + $makemovie + $makeavg + $makeavgmerge + $makeavgplot ))
if [ $make1d2dormovie -gt 0 ]
then

    echo "Doing makemovie.sh stuff"

    for thedir in $dirruns
    do
	echo "Doing makemovielocal for: "$thedir

	cd /data2/jmckinne/${thedir}/$moviedirname

    #################
    if [ $docleanexist -eq 1 ]
    then
        
        echo "clean: "$thedir
        # only clean what one is redoing and isn't fully overwritten
        if [ $make1d -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/qty2_[0-9]*_[0-9]*.npy
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.stderr.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.full.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.out
        fi
        if [ $makemerge -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/qty2.npy
        fi
        if [ $makeplot -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/*.pdf
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python.plot.stderr.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python.plot.full.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python.plot.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/aphi.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/aphi.pdf
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/aphi.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/datavsr*.txt
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/datavsh*.txt
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/datavst*.txt
        fi
        #
        if [ $makemontage -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/montage*.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/montage*.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/montage*.png
        fi
        if [ $makepowervsmplots -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/powervsm*.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/powervsm*.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/powervsm*.png
        fi
        if [ $makespacetimeplots -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/plot*.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/plot*.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/plot*.png
        fi
        if [ $makefftplot -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/fft?.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/fft?.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/fft?.png
        fi
        if [ $makespecplot -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/spec?.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/spec?.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/spec?.png
        fi
        if [ $makeinitfinalplot -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/init1.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/init1.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/init1.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/middle1.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/middle1.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/middle1.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/final1.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/final1.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/final1.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/init1_stream.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/init1_stream.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/init1_stream.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/middle1_stream.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/middle1_stream.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/middle1_stream.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/final1_stream.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/final1_stream.eps
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/final1_stream.png
        fi
        if [ $makethradfinalplot -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/plot0qvsth_.png
        fi
        #
        #
        if [ $makeframes -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.movieframes.stderr.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.movieframes.full.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.movieframes.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/*.png
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/*.eps
        fi
        if [ $makemovie -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/*.avi
        fi
        if [ $makeavg -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.stderr.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.full.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.out
        fi
        if [ $makeavg -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/avg2d[0-9]*_[0-9]*.npy
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.stderr.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.full.out
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/python_[0-9]*_[0-9]*.avg.out
        fi
        if [ $makeavgmerge -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/avg2d[0-9]*_[0-9]*_[0-9]*.npy
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/avg2d.npy
        fi
        if [ $makeavgplot -eq 1 ]
        then
            rm -rf /data2/jmckinne/${thedir}/$moviedirname/fig2.png
        fi
    fi
    ###############

	
	sh makemovielocal.sh ${thedir} $make1d $makemerge $makeplot $makemontage $makepowervsmplots $makespacetimeplots $makefftplot $makespecplot $makeinitfinalplot $makethradfinalplot $makeframes $makemovie $makeavg $makeavgmerge $makeavgplot
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
		    echo "HLatex: \hline" >> tables$moviedirname.tex
        fi
		cat /data2/jmckinne/${thedir}/$moviedirname/python.plot.out | grep "VLatex" >> tables$moviedirname.tex

        iiter=$(( $iiter+1))
    done


    # temporary fix to model names:
    #cat tables$moviedirname.tex | sed 's/0_C/0\_C/g' | sed 's/A94BfN40\\_C5/A-94BfN10\\_Cx/g' | sed 's/A-94BfN10\\_C1/A94BfN40\\_C5/g' | sed 's/A-94BfN10\\_Cx/A-94BfN10\\_C1/g' | sed 's/MB09_D /MB09\\_D /g'  > tables$moviedirname.tex.tmp
    #mv tables$moviedirname.tex.tmp tables$moviedirname.tex

    ##############################################
    #
    # Tables:
    numtbls=17

    for numtbl in `seq 1 $numtbls`
    do
	    echo "Doing Table #: "$numtbl


        ###############################
        fname=table$numtbl$moviedirname.tex
        rm -rf $fname
        #
        echo "\begin{table*}" >> $fname
        if [ $numtbl -eq 1 ]
        then
            echo "\caption{Physical Model Parameters}" >> $fname
        fi
        if [ $numtbl -eq 2 ]
        then
            echo "\caption{Numerical Model Parameters}" >> $fname
        fi
        if [ $numtbl -eq 3 ]
        then
            echo "\caption{Grid Cells across Half-Thickness at Horizon, Half-Thicknesses of Disk, and Location for Interfaces for Disk-Corona and Corona-Jet}" >> $fname
        fi
        if [ $numtbl -eq 16 ]
        then
            echo "\caption{Grid Cells across Half-Thickness at Horizon, Half-Thicknesses of Disk, and Location for Interfaces for Disk-Corona and Corona-Jet}" >> $fname
        fi
        if [ $numtbl -eq 17 ]
        then
            echo "\caption{Grid Cells across Half-Thickness at Horizon, Half-Thicknesses of Disk, and Location for Interfaces for Disk-Corona and Corona-Jet}" >> $fname
        fi
        if [ $numtbl -eq 4 ]
        then
            echo "\caption{Viscosities, Grid Cells per Correlation length and MRI Wavelengths, MRI Wavelengths per full Disk Height, and Radii for MRI Suppression}" >> $fname
        fi
        if [ $numtbl -eq 5 ]
        then
            echo "\caption{Rest-Mass Accretion and Ejection Rates}" >> $fname
        fi
        if [ $numtbl -eq 6 ]
        then
            echo "\caption{Percent Energy Efficiency: BH, Jet, Winds, and NT}" >> $fname
        fi
        if [ $numtbl -eq 7 ]
        then
            echo "\caption{Percent Energy Efficiency: Magnetized Wind and Entire Wind}" >> $fname
        fi
        if [ $numtbl -eq 8 ]
        then
            echo "\caption{Specific Angular Momentum: BH, Jet, Winds, and NT}" >> $fname
        fi
        if [ $numtbl -eq 9 ]
        then
            echo "\caption{Specific Angular Momentum: Magnetized Wind and Entire Wind}" >> $fname
        fi
        if [ $numtbl -eq 10 ]
        then
            echo "\caption{Spin-Up Parameter}" >> $fname
        fi
        if [ $numtbl -eq 15 ]
        then
            echo "\caption{Spin-Up Parameter: BH, Jet, Winds, and NT}" >> $fname
        fi
        if [ $numtbl -eq 11 ]
        then
            echo "\caption{Absolute Magnetic Flux per Rest-Mass Flux and Initial Magnetic Fluxes}" >> $fname
        fi
        if [ $numtbl -eq 12 ]
        then
            echo "\caption{Inner and Outer Radii for Least-Square Fits, Disk+Corona Stagnation Radius, and Fitted Power-Law Indices for Disk Flow}" >> $fname
        fi
        #
        if [ $numtbl -eq 13 ]
        then
            echo "\caption{Inner and Outer Radii for Least-Square Fits, and Fitted Power-Law Indices for Wind Flow}" >> $fname
        fi
        #
        if [ $numtbl -eq 14 ]
        then
            echo "\caption{Inner and Outer Radii for Least-Square Fits, Disk+Corona Stagnation Radius, and Fitted Power-Law Indices for Disk and Wind Flows}" >> $fname
        fi
        #
        echo "\begin{center}" >> $fname
        rawnumc=`grep "Latex$numtbl:" tables$moviedirname.tex | sed 's/[HV]Latex$numtbl: //g' | tail -1 | wc | awk '{print $2}'`
        numc=$(( ($rawnumc - 2)/2 ))
        str1="\begin{tabular}[h]{|"
        str2=""
        for striter in `seq 1 $numc`
        do
            if [ $striter -eq 1 ]
            then
                str2=$str2"l|"
            else
                str2=$str2"r|"
            fi
        done
        str3="}"
        strfinal=$str1$str2$str3
        echo $strfinal >> $fname
        echo "\hline" >> $fname
        # if change model names, probably have to change the below
        #
        if [ $numtbl -eq 14 ] # fits
        then
            # no 2D or MB09D models here
            egrep "Latex$numtbl:|Latex:" tables$moviedirname.tex | sed 's/\([0-9]\)%/\1\\%/g' | sed 's/[HV]Latex'$numtbl': //g' | sed 's/[HV]Latex: //g' | sed 's/\$\&/$ \&/g'   | sed 's/A0\.94BpN100 /\\\\\nA0\.94BpN100 /g' | sed 's/A-0\.94BfN30 /\\\\\nA-0\.94BfN30 /g' | sed 's/A-0\.94BtN10 /\\\\\nA-0\.94BtN10 /g'  | sed 's/MB09Q /\\\\\nMB09Q /g'| sed 's/A-0.9N100 /\\\\\nA-0.9N100 /g'  | sed 's/} \&/}$ \&/g' | sed 's/} \\/}$  \\/g' | sed 's/nan/0/g' | sed 's/e+0/e/g' | sed 's/e-0/e-/g'  | column  -t >> $fname
        else
            egrep "Latex$numtbl:|Latex:" tables$moviedirname.tex | sed 's/\([0-9]\)%/\1\\%/g' | sed 's/[HV]Latex'$numtbl': //g' | sed 's/[HV]Latex: //g' | sed 's/\$\&/$ \&/g'   | sed 's/A0\.94BpN100 /\\\\\nA0\.94BpN100 /g' | sed 's/A-0\.94BfN30 /\\\\\nA-0\.94BfN30 /g' | sed 's/A-0\.94BtN10 /\\\\\nA-0\.94BtN10 /g'  | sed 's/MB09D /\\\\\nMB09D /g'| sed 's/A-0.9N100 /\\\\\nA-0.9N100 /g'  | sed 's/} \&/}$ \&/g' | sed 's/} \\/}$  \\/g' | sed 's/nan/0/g' | sed 's/e+0/e/g' | sed 's/e-0/e-/g'  | column  -t >> $fname
        fi
        #
        echo "\hline" >> $fname
        echo "\hline" >> $fname
        echo "\end{tabular}" >> $fname
        echo "\end{center}" >> $fname
        echo "\label{tbl$numtbl}" >> $fname
        echo "\end{table*}" >> $fname
        ###############################

        # Copy over to final table file names

        cp $fname tbl$numtbl.tex

    done


    echo "For paper, now do:   scp tbl[0-9].tex jon@ki-rh42:/data/jon/thickdisk/harm_thickdisk/ ; scp tbl[0-9][0-9].tex jon@ki-rh42:/data/jon/thickdisk/harm_thickdisk/"
     
    



    ########################
    # Aux tables:

	echo "Doing Aux Tables"

    grep "Latex42:" tables$moviedirname.tex | sed 's/[HV]Latex42: //g'  | column  -t > table42$moviedirname.tex
    grep "Latex93:" tables$moviedirname.tex | sed 's/[HV]Latex93: //g'  | column  -t > table93$moviedirname.tex
    grep "Latex94:" tables$moviedirname.tex | sed 's/[HV]Latex94: //g'  | column  -t > table94$moviedirname.tex
    grep "Latex95:" tables$moviedirname.tex | sed 's/[HV]Latex95: //g'  | column  -t > table95$moviedirname.tex
    grep "Latex96:" tables$moviedirname.tex | sed 's/[HV]Latex96: //g'  | column  -t > table96$moviedirname.tex
    grep "Latex97:" tables$moviedirname.tex | sed 's/[HV]Latex97: //g'  | column  -t > table97$moviedirname.tex
    grep "Latex99:" tables$moviedirname.tex | sed 's/[HV]Latex99: //g'  | column  -t > table99$moviedirname.tex

    echo "Done with collection"

fi

    


echo "Done with all stages"
