#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things


EXPECTED_ARGS=12
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {modelname make1d makemerge makeplot makemontage makepowervsmplots makespacetimeplots makeframes makemovie makeavg makeavgmerge makeavgplot}"
    echo "e.g. sh makemovie.sh thickdisk7 1 1 1 1 1 1 1 0 0 0 0"
    exit $E_BADARGS
fi


modelname=$1
make1d=$2
makemerge=$3
makeplot=$4
makemontage=$5
makepowervsmplots=$6
makespacetimeplots=${7}
makeframes=$8
makemovie=$9
makeavg=${10}
makeavgmerge=${11}
makeavgplot=${12}


jobprefix=$modelname
parallel=0
testrun=0
rminitfiles=0

# 1 = orange
# 2 = orange-gpu
# 3 = ki-jmck
system=3

# can run just certain runi values
useoverride=0
ilistoverride=`seq 24 31`
runnoverride=128

jobsuffix="jy$system"

# runn is number of runs (and in parallel, should be multiple of numcores)

# orange
if [ $system -eq 1 ]
then
    # up to 768 cores (96*2*4)
    # but only 4GB/core.  Seems to work, but maybe much slower than would be if used 6 cores to allow 5.3G/core?
    # ok, now need 5 cores
    numcores=5
    numnodes=36 # so 180 cores total
    thequeue="kipac-ibq"
    # first part of name gets truncated, so use suffix instead for reliability no matter how long the names are
fi

# orange-gpu
if [ $system -eq 2 ]
then
    numcores=8
    numnodes=3
    # there are 16 cores, but have to use 8 for kipac-gpuq too since only 48GB memory and would use up to 5.3GB/core
    # only 3 free nodes for kipac-gpuq (rather than 4 since one is head node)
    thequeue="kipac-gpuq"
fi


# ki-jmck
if [ $system -eq 3 ]
then
    # 4 for thickdisk7 (until new memory put in)
    numcores=12
    numnodes=1
    thequeue="none"
fi


if [ $useoverride -eq 0 ]
then
    runnglobal=$(( $numcores * $numnodes ))
else
    runnglobal=${runnoverride}
fi
echo "runnglobal=$runnglobal"


# for orange systems:
#http://kipac.stanford.edu/collab/computing/hardware/orange/overview
# 96 cnodes, 2 CPUs, 4cores/CPU = 768 cores.
#http://kipac.stanford.edu/collab/computing/docs/orange
#bqueues | grep kipac 
#kipac-ibq       125  Open:Active       -    -    1    -   254     0   254     0
#kipac-gpuq      122  Open:Active       -    -    -    -     0     0     0     0
#kipac-xocmpiq   121  Open:Active       -    -    -    -     0     0     0     0
#kipac-xocq      120  Open:Active       -    -    -    -    41     5    36     0
#kipac-testq      62  Open:Active       -    -    1    -     0     0     0     0
#kipac-ibidleq    15  Open:Active       -    -    1    -     0     0     0     0
#kipac-xocguestq  10  Open:Active       -    -    -    -     0     0     0     0
#

# If you want the movie to contain the bottom panel with Mdot
# vs. time, etc. you need to pre-generate the file, which I call
# qty2.npy: This file contains 1d information for every frame.  I
# generate qty2.npy by running generate_time_series().  


# 1) ensure binary files in place

# see scripts/createlinks.sh

# Sasha: I don't think you need to modify anything (unless you changed
# the meaning of columns in output files or added extra columns to
# fieldline files which would be interpreted by my script as 3 extra
# columns which I added to output gdetB^i's).

# The streamline code does not need gdet B^i's.  It can use B^i's.

# Requirements:
# A) Currently python script requires fieldline0000.bin to exist for getting parameters.  Can search/replace this name for another that actually exists if don't want to include that first file.

# Options: A)
#
# mklotsopanels() has hard-coded indexes of field line files that it
# shows in different panels.  What you see is python complaining it
# cannot find an element with index ti, which is one of the 4 indices
# defined above, findexlist=(0,600,1225,1369) I presume you have less
# than 1369 frames, hence there are problem.  Try resetting findexlist
# to (0,1,2,3), and hopefully the macro will work.

# Options: B)
#
#  In order to run mkstreamlinefigure(), you need to have generated 2D
# average dump files beforehand (search for 2DAVG), created a combined
# average file (for which you run the script with 2DAVG section
# enabled, generate 2D average files, and then run the same script
# again with 3 arguments: start end step, where start and end set the
# starting and ending number of fieldline file group that you average
# over, and each group by default contains 20 fieldline files) and
# named it avg2d.npy .
#
#We can talk more about what you need to do in order to run each of
#the sections of the file.  I have not put any time into making those
#sections portable: as I explained, the code is quite raw!  You might
#want to familiarize yourself with the tutorial first (did it work out
#for you in the end?) and basic python operations.  I am afraid, in
#order to figure out what is going on the script file, you have to be
#able to read python and see what is done.
#
#One thing that can help you, is to enable python debugger.  For this, you run from inside ipython shell:
#
#pdb
#
#Then, whenever there is an error, you get into debugger, where you can evaluate variables, make plots, etc.

# 2) Change "False" to "True" in the section of __main__ that runs
# generate_time_series()

MYPYTHONPATH=$HOME/py/

MREADPATH=$MYPYTHONPATH/mread/
initfile=$MREADPATH/__init__.py
echo "initfile=$initfile"
myrand=${RANDOM}
echo "RANDOM=$myrand"

localpath=`pwd`

passpart1="a#"
passpart2="hyq#ng9"

echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


############################
#
# Make time series (vs. t and vs. r)
#
############################
if [ $make1d -eq 1 ]
then

    runn=${runnglobal}
    echo "runn,runnglobal=$runn $runnglobal"
    numparts=1


    myinitfile1=$localpath/__init__.py.1.$myrand
    echo "myinitfile1="${myinitfile1}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile1


    # 3) already be in <directory that contains "dumps" directory> or cd to it
    
    # can also create new directory and create reduced list of fieldline files.  See createlinksalt.sh
    
    # 4) Then generate the file
    
    # This is a poor-man's parallelization technique: first, thread #i
    # generates a file, qty_${runi}_${runn}.npy, which contains a fraction of
    # time slices.  Then, I merge all these partial files into one single
    # file.
    
    # Requirements to consider inside __init__.py:
    # A) Must have at least one fieldline????.bin dump beyond fti=8000 for averaging period or else script dies.
    # or create a file titf.txt that contains the following:
##comment
#1000 2000 8000 20000
#   last two numbers indicate range of averaging.  E.g., Set 8000->0 if only using fieldline0000.bin dump and no other dumps.

    # Options to consider inside __init__.py:
    # A) 
    
    je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    itot=$(( $runn/$numparts ))
    echo "itot,runn,numparts: $itot $runn $numparts"
    ie=$(( $itot -1 ))

    resid=$(( $runn - $itot * $numparts ))
    
    echo "Running with $itot cores simultaneously"
    
    # just a test:
    # echo "nohup python $myinitfile1 $modelname $runi $runn &> python_${runi}_${runn}.out &"
    # exit 
    
    # LOOP:
    
    for j in `seq 0 $numparts`
    do

	    if [ $j -eq $numparts ]
	    then
	        if [ $resid -gt 0 ]
	        then
		        residie=$(( $resid - 1 ))
		        ilist=`seq 0 $residie`
		        doilist=1
	        else
		        doilist=0
	        fi
	    else
            if [ $useoverride -eq 1 ]
            then
                ilist=$ilistoverride
            else
	            ilist=`seq 0 $ie`
            fi
	        doilist=1
	    fi

	    if [ $doilist -eq 1 ]
	    then
            echo "Data vs. Time: Starting simultaneous run of $itot jobs for group $j"
            for i in $ilist
	        do

	      ############################################################
	      ############# BEGIN WITH RUN IN PARALLEL OR NOT

              # for parallel -eq 1, do every numcores starting with i=0
	            modi=$(($i % $numcores))

	            dowrite=0
	            #if [ $parallel -eq 0 ]
		        #then
		        #    dowrite=1
	            #else
		            if [ $modi -eq 0 ]
		            then
		                dowrite=1
		            fi
	            #fi

	            if [ $dowrite -eq 1 ]
		        then
                  # create script to be run
		            thebatch="sh1_python_${i}_${numcores}_${runn}.sh"
		            rm -rf $thebatch
		            echo "j=$j" >> $thebatch
		            echo "itot=$itot" >> $thebatch
		            echo "i=$i" >> $thebatch
		            echo "runn=$runn" >> $thebatch
		            echo "numcores=$numcores" >> $thebatch
	            fi
	            
	            #if [ $parallel -eq 0 ]
		        #then
		        #    myruni='$(( $i + $itot * $j ))'
		        #    echo "cor=0" >> $thebatch
	            #else
		            echo "i=$i numcores=$numcores modi=$modi"
		            if [ $modi -eq 0 ]
		            then
 		                myruni='$(( $cor - 1 + $i + $itot * $j ))'
		                myseq='`seq 1 $numcores`'
		                if [ $dowrite -eq 1 ]
			            then
			                echo "for cor in $myseq" >> $thebatch
			                echo "do" >> $thebatch
		                fi
		            fi
	            #fi

	            
	            if [ $dowrite -eq 1 ]
		        then
		            echo "runi=$myruni" >> $thebatch
		            echo "textrun=\"Data vs. Time: Running i=\$i j=\$j giving runi=\$runi with runn=\$runn\"" >> $thebatch
		            echo "echo \$textrun" >> $thebatch
		            echo "sleep 1" >> $thebatch
		            cmdraw="python $myinitfile1 $modelname "'$runi $runn'
		            cmdfull='((nohup $cmdraw 2>&1 1>&3 | tee python_${runi}_${cor}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runi}_${cor}_${runn}.out) > python_${runi}_${cor}_${runn}.full.out 2>&1'
		            echo "cmdraw=\"$cmdraw\"" >> $thebatch
		            echo "cmdfull=\"$cmdfull\"" >> $thebatch
		            echo "echo \"\$cmdfull\" > torun_$thebatch.\$cor.sh" >> $thebatch
		            echo "nohup sh torun_$thebatch.\$cor.sh &" >> $thebatch
	            fi


	            #if [ $parallel -eq 1 ]
		        #then
		            if [ $dowrite -eq 1 ]
		            then
		                echo "done" >> $thebatch
		            fi
	            #fi

	            if [ $dowrite -eq 1 ]
		        then
		            echo "wait" >> $thebatch
		            chmod a+x $thebatch
	            fi
	      #
	            if [ $parallel -eq 0 ]
		        then
		            if [ $dowrite -eq 1 ]
		            then
 		                if [ $testrun -eq 1 ]
		                then
		                    echo $thebatch
		                else
		                    sh ./$thebatch
		                fi
                    fi
	            else
		            if [ $dowrite -eq 1 ]
		            then
    	              # run bsub on batch file
		                jobname=$jobprefix.${i}.$jobsuffix
		                outputfile=$jobname.out
		                errorfile=$jobname.err
              # probably specifying ptile below is not necessary
		                bsubcommand="bsub -n 1 -x -R span[ptile=$numcores] -q $thequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
		                if [ $testrun -eq 1 ]
			            then
			                echo $bsubcommand
		                else
			                echo $bsubcommand
			                echo "$bsubcommand" > bsubshtorun_$thebatch
			                chmod a+x bsubshtorun_$thebatch
			                sh bsubshtorun_$thebatch
		                fi
		            fi
	            fi

	      ############# END WITH RUN IN PARALLEL OR NOT
	      ############################################################
	            
	        done

	    # waiting game
	        if [ $parallel -eq 0 ]
		    then
		        wait
	        else
                # Wait to move on until all jobs done
		        totaljobs=$runn
		        firsttimejobscheck=1
		        while [ $totaljobs -gt 0 ]
		        do
		            pendjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobsuffix | grep jmckinn | grep PEND | wc -l`
		            runjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobsuffix | grep jmckinn | grep RUN | wc -l`
		            totaljobs=$(($pendjobs+$runjobs))
		            
		            if [ $totaljobs -gt 0 ]
		            then
		                echo "PEND=$pendjobs RUN=$runjobs TOTAL=$totaljobs ... waiting ..."
		                sleep 10
                        firsttimejobscheck=0
		            else
		                if [ $firsttimejobscheck -eq 1 ]
			            then
			                totaljobs=$runn
			                echo "waiting for jobs to get started..."
			                sleep 10
		                else
			                echo "DONE!"		      
		                fi
		            fi
		        done

	        fi


	        echo "Data vs. Time: Ending simultaneous run of $itot jobs for group $j"
	    fi
    done
    
    wait

    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile1
    fi

fi


echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# Merge npy files
#
####################################
if [ $makemerge -eq 1 ]
then

    # runn should be same as when creating files
    runn=${runnglobal}


    myinitfile2=$localpath/__init__.py.2.$myrand
    echo "myinitfile2="${myinitfile2}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile2


    echo "Merge to single file"
    if [ $testrun -eq 1 ]
	then
	    echo "((nohup python $myinitfile2 $modelname $runn $runn 2>&1 1>&3 | tee python_${runn}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.out) > python_${runn}_${runn}.full.out 2>&1"
    else
	    ((nohup python $myinitfile2 $modelname $runn $runn 2>&1 1>&3 | tee python_${runn}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.out) > python_${runn}_${runn}.full.out 2>&1
    fi

    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile2
    fi

fi


echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# Maake plots and latex tables
#
####################################
if [ $makeplot -eq 1 ]
then

    myinitfile3=$localpath/__init__.py.3.$myrand
    echo "myinitfile3="${myinitfile3}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile3


    echo "Generate the plots"
    # &> 
    if [ $testrun -eq 1 ]
	then
	    echo "((nohup python $myinitfile3 $modelname 2>&1 1>&3 | tee python.plot.stderr.out) 3>&1 1>&2 | tee python.plot.out) > python.plot.full.out 2>&1"
    else
        # string "plot" tells script to do plot
	    ((nohup python $myinitfile3 $modelname plot $makepowervsmplots $makespacetimeplots 2>&1 1>&3 | tee python.plot.stderr.out) 3>&1 1>&2 | tee python.plot.out) > python.plot.full.out 2>&1
    fi

    makemontage=$makemontage
    if [ $makemontage -eq 1 ]
    then
        # create montage of t vs. r and t vs. h plots
        files=`ls -rt plot*.png`
        montage -geometry 300x600 $files montage_plot.png
        # use -density to control each image size
        # e.g. default for montage is:
        # montage -density 72 $files montage.png
        # but can get each image as 300x300 (each tile) if do:
        # montage -geometry 300x300 -density 300 $files montage.png
        #
        # to display, do:
        # display montage.png
        # if want to have smaller window and pan more do (e.g.):
        # display -geometry 1800x1400 montage.png

        files=`ls -rt powervsm*.png`
        montage -geometry 500x500 $files montage_powervsm.png

    fi


    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile3
    fi

fi



echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# MOVIE FRAMES
#
####################################
if [ $makeframes -eq 1 ]
then



    # Now you are ready to generate movie frames, you can do that in
    # parallel, too, in a very similar way.
    
    # 1) You disable the time series section of ~/py/mread/__init__.py and
    # instead enable the movie section
    
    myinitfile4=$localpath/__init__.py.${myrand}.4
    echo "myinitfile4="${myinitfile4}
    
    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#make a movie[\n \t]*mkmovie()/if True:\n\t#make a movie\n\tmkmovie()/g;p;}'  $initfile > $myinitfile4
    
    
    # Options to consider inside __init__.py:
    #
    # A) can change showextra=False to True:
    # def plotqtyvstime(qtymem,ihor=11,whichplot=None,ax=None,findex=None,fti=None,ftf=None,showextra=True,prefactor=100)
    # 
    #
    # B) Can choose vmin and vmax for lrho range in movie:
    # mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False)
    # mkframexy("lrho%04d_xy%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax2,cb=True,pt=False,dostreamlines=True)
    #
    # C) Can choose frame size by changing plotlen=# where # is # of M that plot will go to in each direction.  Change plotlen when plotgen being setup in mkmovie().  Or directly change mkmovie(framesize=50) to another #.


    
    # 2) Now run job as before.  But makeing movie frames takes about 2X more memory, so increase parts by 2X
    
    runn=${runnglobal}
    numparts=1

    
    je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    itot=$(( $runn/$numparts ))
    ie=$(( $itot -1 ))
    
    echo "Running with $itot cores simultaneously"
    
    
    for j in `seq 0 $numparts`
    do


	    if [ $j -eq $numparts ]
	    then
	        if [ $resid -gt 0 ]
	        then
		        residie=$(( $resid - 1 ))
		        ilist=`seq 0 $residie`
		        doilist=1
	        else
		        doilist=0
	        fi
	    else
	        ilist=`seq 0 $ie`
	        doilist=1
	    fi

	    if [ $doilist -eq 1 ]
	    then
	        echo "Movie Frames: Starting simultaneous run of $itot jobs for group $j"
	        for i in $ilist
	        do

	      ############################################################
	      ############# BEGIN WITH RUN IN PARALLEL OR NOT

              # for parallel -eq 1, do every numcores starting with i=0
	            modi=$(($i % $numcores))

	            dowrite=0
	            #if [ $parallel -eq 0 ]
		        #then
		        #    dowrite=1
	            #else
		            if [ $modi -eq 0 ]
		            then
		                dowrite=1
		            fi
	            #fi

	            if [ $dowrite -eq 1 ]
		        then
                  # create script to be run
		            thebatch="sh4_python_${i}_${numcores}_${runn}.sh"
		            rm -rf $thebatch
		            echo "j=$j" >> $thebatch
		            echo "itot=$itot" >> $thebatch
		            echo "i=$i" >> $thebatch
		            echo "runn=$runn" >> $thebatch
		            echo "numcores=$numcores" >> $thebatch
	            fi
	            
	            #if [ $parallel -eq 0 ]
		        #then
		        #    myruni='$(( $i + $itot * $j ))'
		        #    echo "cor=0" >> $thebatch
	            #else
		            echo "i=$i numcores=$numcores modi=$modi"
		            if [ $modi -eq 0 ]
		            then
 		                myruni='$(( $cor - 1 + $i + $itot * $j ))'
		                myseq='`seq 1 $numcores`'
		                if [ $dowrite -eq 1 ]
			            then
			                echo "for cor in $myseq" >> $thebatch
			                echo "do" >> $thebatch
		                fi
		            fi
	            #fi

	            
	            if [ $dowrite -eq 1 ]
		        then
		            echo "runi=$myruni" >> $thebatch
		            echo "textrun=\"Movie Frames vs. Time: Running i=\$i j=\$j giving runi=\$runi with runn=\$runn\"" >> $thebatch
		            echo "echo \$textrun" >> $thebatch
		            echo "sleep 1" >> $thebatch
		            cmdraw="python $myinitfile4 $modelname "'$runi $runn'
		            cmdfull='((nohup $cmdraw 2>&1 1>&3 | tee python_${runi}_${cor}_${runn}.stderr.movieframes.out) 3>&1 1>&2 | tee python_${runi}_${cor}_${runn}.movieframes.out) > python_${runi}_${cor}_${runn}.full.movieframes.out 2>&1'
		            echo "cmdraw=\"$cmdraw\"" >> $thebatch
		            echo "cmdfull=\"$cmdfull\"" >> $thebatch
		            echo "echo \"\$cmdfull\" > torun_$thebatch.\$cor.sh" >> $thebatch
		            echo "nohup sh torun_$thebatch.\$cor.sh &" >> $thebatch
	            fi


	            #if [ $parallel -eq 1 ]
		        #then
		            if [ $dowrite -eq 1 ]
		            then
		                echo "done" >> $thebatch
		            fi
	            #fi

	            if [ $dowrite -eq 1 ]
		        then
		            echo "wait" >> $thebatch
		            chmod a+x $thebatch
	            fi
	      #
	            if [ $parallel -eq 0 ]
		        then
		            if [ $dowrite -eq 1 ]
		            then
		                if [ $testrun -eq 1 ]
		                then
		                    echo $thebatch
		                else
		                    echo $thebatch
		                    sh ./$thebatch
		                fi
                    fi
	            else
		            if [ $dowrite -eq 1 ]
		            then
    	              # run bsub on batch file
		                jobname=$jobprefix.${i}.$jobsuffix
		                outputfile=$jobname.out
		                errorfile=$jobname.err
              # probably specifying ptile below is not necessary
		                bsubcommand="bsub -n 1 -x -R span[ptile=$numcores] -q $thequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
		                if [ $testrun -eq 1 ]
			            then
			                echo $bsubcommand
		                else
			                echo $bsubcommand
			                echo "$bsubcommand" > bsubshtorun_$thebatch
			                chmod a+x bsubshtorun_$thebatch
			                sh bsubshtorun_$thebatch
		                fi
		            fi
	            fi

	      ############# END WITH RUN IN PARALLEL OR NOT
	      ############################################################

	        done

	    # waiting game
	        if [ $parallel -eq 0 ]
		    then
		        wait
	        else
                # Wait to move on until all jobs done
		        totaljobs=$runn
		        firsttimejobscheck=1
		        while [ $totaljobs -gt 0 ]
		        do
		            pendjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobsuffix | grep jmckinn | grep PEND | wc -l`
		            runjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobsuffix | grep jmckinn | grep RUN | wc -l`
		            totaljobs=$(($pendjobs+$runjobs))
		            
		            if [ $totaljobs -gt 0 ]
		            then
		                echo "PEND=$pendjobs RUN=$runjobs TOTAL=$totaljobs ... waiting ..."
		                sleep 10
                        firsttimejobscheck=0
		            else
		                if [ $firsttimejobscheck -eq 1 ]
			            then
			                totaljobs=$runn
			                echo "waiting for jobs to get started..."
			                sleep 10
		                else
			                echo "DONE!"		      
		                fi
		            fi
		        done

	        fi

	        echo "Movie Frames: Ending simultaneous run of $itot jobs for group $j"
	    fi
    done
    
    
    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile4
    fi
    
fi

echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# MOVIE File
#
####################################
if [ $makemovie -eq 1 ]
then

    #  now can create an avi with:
    
    if [ $testrun -eq 1 ]
	then
	    echo "make movie files"
    else
        
        
        fps=25
        #
        #ffmpeg -i lrho%04d_Rzxym1.png -r $fps -sameq lrho.mp4
        #ffmpeg -fflags +genpts -i lrho%04d_Rzxym1.png -r $fps -sameq lrho.$modelname.avi

	    if [ 1 -eq 0 ]
	    then
        # high quality 1 minute long no matter what framerate (-t 60 doesn't work)
	        ffmpeg -y -fflags +genpts -i lrho%04d_Rzxym1.png -r 25 -sameq -qmax 5 -vcodec mjpeg lrho25.$modelname.avi 
        # now set frame rate (changes duration)
	        ffmpeg -y -i lrho25.$modelname.avi -f image2pipe -vcodec copy - </dev/null | ffmpeg -r $fps -f image2pipe -vcodec mjpeg -i - -vcodec copy -an lrho.$modelname.avi
	        
        # high quality 1 minute long no matter what framerate (-t 60 doesn't work)
	        ffmpeg -y -fflags +genpts -i lrhosmall%04d_Rzxym1.png -r 25 -sameq -qmax 5 -vcodec mjpeg lrhosmall25.$modelname.avi 
        # now set frame rate (changes duration)
	        ffmpeg -y -i lrhosmall25.$modelname.avi -f image2pipe -vcodec copy - </dev/null | ffmpeg -r $fps -f image2pipe -vcodec mjpeg -i - -vcodec copy -an lrhosmall.$modelname.avi
	    else
        # Sasha's command:
	        ffmpeg -y -fflags +genpts -r $fps -i lrho%04d_Rzxym1.png -vcodec mpeg4 -sameq -qmax 5 lrho.$modelname.avi
	        ffmpeg -y -fflags +genpts -r $fps -i lrhosmall%04d_Rzxym1.png -vcodec mpeg4 -sameq -qmax 5 lrhosmall.$modelname.avi
	        
        # for Roger (i.e. any MAC)
	        ffmpeg -y -fflags +genpts -r $fps -i lrho%04d_Rzxym1.png -sameq -qmax 5 lrho.$modelname.mov
	        ffmpeg -y -fflags +genpts -r $fps -i lrhosmall%04d_Rzxym1.png -sameq -qmax 5 lrhosmall.$modelname.mov
	    fi
    fi

    echo "Now do: mplayer -loop 0 lrho.$modelname.avi OR lrhosmall.$modelname.avi"

fi


echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# avg File (takes average of 20 fieldline files per avg file created)
#
####################################
if [ $makeavg -eq 1 ]
then

    echo "Doing avg file"

    runn=${runnglobal}
    numparts=1


    myinitfile5=$localpath/__init__.py.5.$myrand
    echo "myinitfile5="${myinitfile5}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#2DAVG[\n \t]*mk2davg()/if True:\n\t#2DAVG\n\tmk2davg()/g;p;}'  $initfile > $myinitfile5
    
    je=$(( $numparts - 1 ))
    # above two must be exactly divisible
    itot=$(( $runn/$numparts ))
    ie=$(( $itot -1 ))

    resid=$(( $runn - $itot * $numparts ))
    
    echo "Running with $itot cores simultaneously"
    
    # just a test:
    # echo "nohup python $myinitfile1 $modelname $runi $runn &> python_${runi}_${runn}.out &"
    # exit 
    
    # LOOP:
    
    for j in `seq 0 $numparts`
    do

	    if [ $j -eq $numparts ]
	    then
	        if [ $resid -gt 0 ]
	        then
		        residie=$(( $resid - 1 ))
		        ilist=`seq 0 $residie`
		        doilist=1
	        else
		        doilist=0
	        fi
	    else
	        ilist=`seq 0 $ie`
	        doilist=1
	    fi

	    if [ $doilist -eq 1 ]
	    then
            echo "Data vs. Time: Starting simultaneous run of $itot jobs for group $j"
            for i in $ilist
            do







	      ############################################################
	      ############# BEGIN WITH RUN IN PARALLEL OR NOT

              # for parallel -eq 1, do every numcores starting with i=0
	            modi=$(($i % $numcores))

	            dowrite=0
	            #if [ $parallel -eq 0 ]
		        #then
		        #    dowrite=1
	            #else
		            if [ $modi -eq 0 ]
		            then
		                dowrite=1
		            fi
	            #fi

	            if [ $dowrite -eq 1 ]
		        then
                  # create script to be run
		            thebatch="sh5_python_${i}_${numcores}_${runn}.sh"
		            rm -rf $thebatch
		            echo "j=$j" >> $thebatch
		            echo "itot=$itot" >> $thebatch
		            echo "i=$i" >> $thebatch
		            echo "runn=$runn" >> $thebatch
		            echo "numcores=$numcores" >> $thebatch
	            fi
	            
	            #if [ $parallel -eq 0 ]
		        #then
		        #    myruni='$(( $i + $itot * $j ))'
		        #    echo "cor=0" >> $thebatch
	            #else
		            echo "i=$i numcores=$numcores modi=$modi"
		            if [ $modi -eq 0 ]
		            then
 		                myruni='$(( $cor - 1 + $i + $itot * $j ))'
		                myseq='`seq 1 $numcores`'
		                if [ $dowrite -eq 1 ]
			            then
			                echo "for cor in $myseq" >> $thebatch
			                echo "do" >> $thebatch
		                fi
		            fi
	            #fi

	            
	            if [ $dowrite -eq 1 ]
		        then
		            echo "runi=$myruni" >> $thebatch
		            echo "textrun=\"Avg Data vs. Time: Running i=\$i j=\$j giving runi=\$runi with runn=\$runn\"" >> $thebatch
		            echo "echo \$textrun" >> $thebatch
		            echo "sleep 1" >> $thebatch
		            cmdraw="python $myinitfile5 $modelname "'$runi $runn'
		            cmdfull='((nohup $cmdraw 2>&1 1>&3 | tee python_${runi}_${cor}_${runn}.stderr.avg.out) 3>&1 1>&2 | tee python_${runi}_${cor}_${runn}.avg.out) > python_${runi}_${cor}_${runn}.full.avg.out 2>&1'
		            echo "cmdraw=\"$cmdraw\"" >> $thebatch
		            echo "cmdfull=\"$cmdfull\"" >> $thebatch
		            echo "echo \"\$cmdfull\" > torun_$thebatch.\$cor.sh" >> $thebatch
		            echo "sh torun_$thebatch.\$cor.sh &" >> $thebatch
	            fi


	            #if [ $parallel -eq 1 ]
		        #then
		            if [ $dowrite -eq 1 ]
		            then
		                echo "done" >> $thebatch
		            fi
	            #fi

	            if [ $dowrite -eq 1 ]
		        then
		            echo "wait" >> $thebatch
		            chmod a+x $thebatch
	            fi
	      #
	            if [ $parallel -eq 0 ]
		        then
		            if [ $dowrite -eq 1 ]
		            then
		                if [ $testrun -eq 1 ]
		                then
		                    echo $thebatch
		                else
		                    echo $thebatch
		                    sh ./$thebatch
		                fi
                    fi
	            else
		            if [ $dowrite -eq 1 ]
		            then
    	              # run bsub on batch file
		                jobname=$jobprefix.${i}.$jobsuffix
		                outputfile=$jobname.out
		                errorfile=$jobname.err
              # probably specifying ptile below is not necessary
		                bsubcommand="bsub -n 1 -x -R span[ptile=$numcores] -q $thequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
		                if [ $testrun -eq 1 ]
			            then
			                echo $bsubcommand
		                else
			                echo $bsubcommand
			                echo "$bsubcommand" > bsubshtorun_$thebatch
			                chmod a+x bsubshtorun_$thebatch
			                sh bsubshtorun_$thebatch
		                fi
		            fi
	            fi

	      ############# END WITH RUN IN PARALLEL OR NOT
	      ############################################################


	        done


	    # waiting game
	        if [ $parallel -eq 0 ]
		    then
		        wait
	        else
                # Wait to move on until all jobs done
		        totaljobs=$runn
		        firsttimejobscheck=1
		        while [ $totaljobs -gt 0 ]
		        do
		            pendjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobsuffix | grep jmckinn | grep PEND | wc -l`
		            runjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobsuffix | grep jmckinn | grep RUN | wc -l`
		            totaljobs=$(($pendjobs+$runjobs))
		            
		            if [ $totaljobs -gt 0 ]
		            then
		                echo "PEND=$pendjobs RUN=$runjobs TOTAL=$totaljobs ... waiting ..."
		                sleep 10
		                firsttimejobscheck=0
		            else
		                if [ $firsttimejobscheck -eq 1 ]
			            then
			                totaljobs=$runn
			                echo "waiting for jobs to get started..."
			                sleep 10
		                else
			                echo "DONE!"		      
		                fi
		            fi
		        done

	        fi

	        echo "Data vs. Time: Ending simultaneous run of $itot jobs for group $j"
	    fi
    done
    
    wait

    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile5
    fi

fi

echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# Merge avg npy files
#
####################################
if [ $makeavgmerge -eq 1 ]
then

    # should be same as when creating avg files
    runn=${runnglobal}


    myinitfile6=$localpath/__init__.py.6.$myrand
    echo "myinitfile6="${myinitfile6}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#2DAVG[\n \t]*mk2davg()/if True:\n\t#2DAVG\n\tmk2davg()/g;p;}'  $initfile > $myinitfile6


    echo "Merge avg files to single avg file"
    # <index of first avg file to use> <index of last avg file to use> <step=1>
    # must be step=1, or no merge occurs
    step=1
    itemspergroup=$(( 20 ))
    whichgroups=$(( 0 ))
    numavg2dmerge=`ls -vrt | egrep "avg2d"${itemspergroup}"_[0-9]*\.npy"|wc|awk '{print $1}'`
    #whichgroupe=$(( $itemspergroup * $runn ))
    whichgroupe=$numavg2dmerge

    groupsnum=`printf "%04d" "$whichgroups"`
    groupenum=`printf "%04d" "$whichgroupe"`

    echo "GROUPINFO: $step $itemspergroup $whichgroups $whichgroupe $groupsnum $groupenum"

    if [ $testrun -eq 1 ]
	then
	    echo "((nohup python $myinitfile6 $modelname $whichgroups $whichgroupe $step 2>&1 1>&3 | tee python_${runn}_${runn}.avg.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.avg.out) > python_${runn}_${runn}.avg.full.out 2>&1"
    else
	    ((nohup python $myinitfile6 $modelname $whichgroups $whichgroupe $step 2>&1 1>&3 | tee python_${runn}_${runn}.avg.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.avg.out) > python_${runn}_${runn}.avg.full.out 2>&1

        # copy resulting avg file to avg2d.npy
	    avg2dmerge=`ls -vrt avg2d${itemspergroup}_${groupsnum}_${groupenum}.npy | head -1`    
	    cp $avg2dmerge avg2d.npy

    fi

    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile6
    fi




fi

echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# Make avg plot
#
####################################
if [ $makeavgplot -eq 1 ]
then

    myinitfile7=$localpath/__init__.py.7.$myrand
    echo "myinitfile7="${myinitfile7}

    sed -n '1h;1!H;${;g;s/if False:[\n \t]*#fig2 with grayscalestreamlines and red field lines[\n \t]*mkstreamlinefigure()/if True:\n\t#fig2 with grayscalestreamlines and red field lines\n\tmkstreamlinefigure()/g;p;}'  $initfile > $myinitfile7


    echo "Generate the avg plots"
    # &> 
    if [ $testrun -eq 1 ]
	then
	    echo "((nohup python $myinitfile7 $modelname 2>&1 1>&3 | tee python.plot.avg.stderr.out) 3>&1 1>&2 | tee python.plot.avg.out) > python.plot.avg.full.out 2>&1"
    else
	    ((nohup python $myinitfile7 $modelname 2>&1 1>&3 | tee python.plot.avg.stderr.out) 3>&1 1>&2 | tee python.plot.avg.out) > python.plot.avg.full.out 2>&1
    fi

    
    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile7
    fi

fi


echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


# to clean-up bad start, use:
# rm -rf sh*.sh bsub*.sh __init* py*.out torun*.sh j1*.err j1*.out *.npy
#
