#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things

# Steps:
#
# 1) Ensure followed createlinks.sh for each directory.
# 2) use the script

# order of models as to appear in final table
dircollect='thickdisk7 thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdiskr15 thickdisk2 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99'

# note that thickdisk1 is actually bad, so ignore it.
# can choose so do runs in different order than collection.
#dirruns=$dircollect
# do expensive thickdisk7 and sasha99 last so can test things
dirruns='thickdisk8 thickdisk11 thickdisk12 thickdisk13 run.like8 thickdiskrr2 run.liker2butbeta40 run.liker2 thickdisk16 thickdisk5 thickdisk14 thickdiskr1 thickdiskr2 run.liker1 thickdisk9 thickdiskr3  thickdisk17 thickdisk10 thickdisk15 thickdiskr15 thickdisk2 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

#dirruns='thickdisk5 thickdiskr3 thickdisk17 thickdisk10 thickdisk15 thickdiskr15 thickdisk2 thickdisk3 thickdiskhr3 runlocaldipole3dfiducial blandford3d_new sasham9 sasham5 sasha0 sasha1 sasha2 sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

#dirruns='sasha5 sasha9b25 sasha9b50 sasha9b100 sasha9b200 sasha99 thickdisk7'

#dirruns='thickdisk7 sasha99'
#dirruns='runlocaldipole3dfiducial'
#dirruns='sasha99'

# number of files to keep
numkeep=300


EXPECTED_ARGS=16
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {moviedirname docleanexist dolinks dofiles make1d makemerge makeplot makemontage makepowervsmplots makespacetimeplots makeframes makemovie makeavg makeavgmerge makeavgplot collect}"
    echo "e.g. sh makeallmovie.sh moviefinal1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
    exit $E_BADARGS
fi


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
makeframes=${11}
makemovie=${12}
makeavg=${13}
makeavgmerge=${14}
makeavgplot=${15}
collect=${16}


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
    if [ "$thedir" == "sasha99" ]
    then
        factor=4
        # it was a long run, but quasi-steady at t=8000 out of tf=32000
    else
        # most steady by 8000 but run till only 13000 for dipoley runs
        # or for toroidal runs, ran for 2X when was steady.  So also good.
        factor=2
    fi
    #
    if [ "$thedir" == "thickdiskhr3" ]
    then
        factor=100000000
    fi
    #    #
    #
    keepfilesstart=$(( (1 * $numfiles) / $factor ))
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


done

    echo "Done with files"


fi




##############################################
make1d2dormovie=$(( $make1d + $makemerge + $makeplot + $makemontage + $makepowervsmplots + $makespacetimeplots + $makeframes + $makemovie + $makeavg + $makeavgmerge + $makeavgplot ))
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

	
	sh makemovielocal.sh ${thedir} $make1d $makemerge $makeplot $makemontage $makepowervsmplots $makespacetimeplots $makeframes $makemovie $makeavg $makeavgmerge $makeavgplot
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

    for numtbl in `seq 1 12`
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
            echo "\caption{Grid Cells across Half-Thickness at Horizon, Half-Thickness of Disk, and Location for Interfaces for Disk-Corona and Corona-Jet}" >> $fname
        fi
        if [ $numtbl -eq 4 ]
        then
            echo "\caption{Magnetic Stress, Grid Cells per MRI Wavelength, and MRI Wavelengths per Disk Height}" >> $fname
        fi
        if [ $numtbl -eq 5 ]
        then
            echo "\caption{Rest-Mass Accretion and Ejection Rates}" >> $fname
        fi
        if [ $numtbl -eq 6 ]
        then
            echo "\caption{Percent Energy Efficiency: BH, Jet+Outflows, Jet, and NT}" >> $fname
        fi
        if [ $numtbl -eq 7 ]
        then
            echo "\caption{Percent Energy Efficiency: Magnetized Wind and Entire Wind}" >> $fname
        fi
        if [ $numtbl -eq 8 ]
        then
            echo "\caption{Specific Angular Momentum: BH, Jet, Totals, and NT}" >> $fname
        fi
        if [ $numtbl -eq 9 ]
        then
            echo "\caption{Specific Angular Momentum: Magnetized Wind and Entire Wind}" >> $fname
        fi
        if [ $numtbl -eq 10 ]
        then
            echo "\caption{Spin-Up Parameter}" >> $fname
        fi
        if [ $numtbl -eq 11 ]
        then
            echo "\caption{Absolute Magnetic Flux per Rest-Mass Flux and Initial Magnetic Fluxes}" >> $fname
        fi
        if [ $numtbl -eq 12 ]
        then
            echo "\caption{Inner and Outer Radii for Least-Square Fits, Disk+Corona Stagnation Radius, and Fitted Power-Law Indices}" >> $fname
        fi
        #
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
        egrep "Latex$numtbl:|Latex:" tables$moviedirname.tex | sed 's/\([0-9]\)%/\1\\%/g' | sed 's/[HV]Latex'$numtbl': //g' | sed 's/[HV]Latex: //g' | sed 's/\$\&/$ \&/g'   | sed 's/A94BpN100 /\\\\\nA94BpN100 /g' | sed 's/A-94BfN30 /\\\\\nA-94BfN30 /g' | sed 's/A-94BtN10 /\\\\\nA-94BtN10 /g'  | sed 's/MB09D /\\\\\nMB09D /g'| sed 's/A-0.9N100 /\\\\\nA-0.9N100 /g'  | sed 's/} \&/}$ \&/g' | sed 's/} \\/}$  \\/g' | sed 's/nan/0/g' | sed 's/e+0/e/g' | sed 's/e-0/e-/g'  | column  -t >> $fname
        echo "\hline" >> $fname
        echo "\hline" >> $fname
        echo "\end{tabular}" >> $fname
        echo "\end{center}" >> $fname
        echo "\label{tbl$numtbl}" >> $fname
        echo "\end{table*}" >> $fname
        ###############################

        # Copy over to final table file names

        cp $fname table$numtbl.tex

    done


    echo "For paper, now do:   scp table[0-9].tex jon@ki-rh42:/data/jon/thickdisk/harm_thickdisk/ ; scp table[0-9][0-9].tex jon@ki-rh42:/data/jon/thickdisk/harm_thickdisk/"
     
    



    ########################
    # Aux tables:

	echo "Doing Aux Tables"

    grep "Latex93:" tables$moviedirname.tex | sed 's/[HV]Latex93: //g'  | column  -t > table93$moviedirname.tex
    grep "Latex94:" tables$moviedirname.tex | sed 's/[HV]Latex94: //g'  | column  -t > table94$moviedirname.tex
    grep "Latex95:" tables$moviedirname.tex | sed 's/[HV]Latex95: //g'  | column  -t > table95$moviedirname.tex
    grep "Latex96:" tables$moviedirname.tex | sed 's/[HV]Latex96: //g'  | column  -t > table96$moviedirname.tex
    grep "Latex97:" tables$moviedirname.tex | sed 's/[HV]Latex97: //g'  | column  -t > table97$moviedirname.tex
    grep "Latex99:" tables$moviedirname.tex | sed 's/[HV]Latex99: //g'  | column  -t > table99$moviedirname.tex

    echo "Done with collection"

fi

    


echo "Done with all stages"
