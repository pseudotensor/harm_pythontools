#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things


###############################################################
# SPECIAL NOTES:
#
# 1) On kraken, have to have source ~/setuppython27 before running makeallmovie.sh or makemovie.sh so that makemoviec compiled properly and python run properly on compute nodes.
#   Can't source here because "module" program doesn't exist in bash
# 
###############################################################



EXPECTED_ARGS=18
E_BADARGS=65

#if [ $# -ne $EXPECTED_ARGS ]
if [ $# -lt $(($EXPECTED_ARGS)) ]
then
    echo "Usage: `basename $0` {modelname make1d makemerge makeplot makemontage makepowervsmplots makespacetimeplots makefftplot makespecplot makeinitfinalplot makethradfinalplot makeframes makemovie makeavg makeavgmerge makeavgplot} <system> <parallel> <dirname>"
    echo "only dirname is optional"
    echo "e.g. sh makemovie.sh thickdisk7 1 1 1 1 1 1 1 0 0 0 0    3 0 /data1/jmckinne/thickdisk7/fulllatest14/"
    exit $E_BADARGS
fi


modelname=$1
make1d=$2
makemerge=$3
makeplot=$4
makemontage=$5
makepowervsmplots=$6
makespacetimeplots=${7}
makefftplot=${8}
makespecplot=${9}
makeinitfinalplot=${10}
makethradfinalplot=${11}
makeframes=${12}
makemovie=${13}
makeavg=${14}
makeavgmerge=${15}
makeavgplot=${16}

system=${17}
parallel=${18}

# get optional dirname
if [ $# -eq $(($EXPECTED_ARGS+1))  ]
then
    dirname=${19}
else
    # assume just local directory if not given
    # should be full path
    dirname=`pwd`
fi



###########################################
#
# parameters one can set
#
###########################################

modelnamelen=${#modelname}
jobprefix=${modelname} #:((modelnamelen-4)):3}
testrun=0
rminitfiles=0


# can run just certain runi values
useoverride=0
ilistoverride=`seq 24 31`
runnoverride=128


#########################################################################
# define unique suffix so know which thing is running in batch system
# below list obtained from __init__.py and then processed for bash
if [ $modelname == "thickdisk7" ]
then
    jobsuffix="jy$system"       
elif [ $modelname == "thickdisk8" ]
then
    jobsuffix="jb$system"       
elif [ $modelname == "thickdisk11" ]
then
    jobsuffix="jc$system"       
elif [ $modelname == "thickdisk12" ]
then
    jobsuffix="jd$system"       
elif [ $modelname == "thickdisk13" ]
then
    jobsuffix="je$system"       
elif [ $modelname == "run.like8" ]
then
    jobsuffix="jf$system"       
elif [ $modelname == "thickdiskrr2" ]
then
    jobsuffix="jg$system"       
elif [ $modelname == "run.liker2butbeta40" ]
then
    jobsuffix="jh$system"       
elif [ $modelname == "run.liker2" ]
then
    jobsuffix="ji$system"       
elif [ $modelname == "thickdisk16" ]
then
    jobsuffix="jj$system"       
elif [ $modelname == "thickdisk5" ]
then
    jobsuffix="jk$system"       
elif [ $modelname == "thickdisk14" ]
then
    jobsuffix="jl$system"       
elif [ $modelname == "thickdiskr1" ]
then
    jobsuffix="jm$system"       
elif [ $modelname == "run.liker1" ]
then
    jobsuffix="jn$system"       
elif [ $modelname == "thickdiskr2" ]
then
    jobsuffix="jo$system"       
elif [ $modelname == "thickdisk9" ]
then
    jobsuffix="jp$system"       
elif [ $modelname == "thickdiskr3" ]
then
    jobsuffix="jq$system"       
elif [ $modelname == "thickdisk17" ]
then
    jobsuffix="jr$system"       
elif [ $modelname == "thickdisk10" ]
then
    jobsuffix="js$system"       
elif [ $modelname == "thickdisk15" ]
then
    jobsuffix="jt$system"       
elif [ $modelname == "thickdiskr15" ]
then
    jobsuffix="ju$system"       
elif [ $modelname == "thickdisk2" ]
then
    jobsuffix="jv$system"       
elif [ $modelname == "thickdisk3" ]
then
    jobsuffix="jw$system"       
elif [ $modelname == "thickdiskhr3" ]
then
    jobsuffix="jx$system"       
elif [ $modelname == "runlocaldipole3dfiducial" ]
then
    jobsuffix="ja$system"       
elif [ $modelname == "blandford3d_new" ]
then
    jobsuffix="ka$system"       
elif [ $modelname == "a0hr07" ]
then
    jobsuffix="kb$system"       
elif [ $modelname == "sasham9" ]
then
    jobsuffix="kc$system"       
elif [ $modelname == "sasham9full2pi" ]
then
    jobsuffix="kd$system"       
elif [ $modelname == "sasham5" ]
then
    jobsuffix="ke$system"       
elif [ $modelname == "sasham2" ]
then
    jobsuffix="kf$system"       
elif [ $modelname == "sasha0" ]
then
    jobsuffix="kg$system"       
elif [ $modelname == "sasha1" ]
then
    jobsuffix="kh$system"       
elif [ $modelname == "sasha2" ]
then
    jobsuffix="ki$system"       
elif [ $modelname == "sasha5" ]
then
    jobsuffix="kj$system"       
elif [ $modelname == "sasha9b25" ]
then
    jobsuffix="kk$system"       
elif [ $modelname == "sasha9b50" ]
then
    jobsuffix="kl$system"       
elif [ $modelname == "sasha9b100" ]
then
    jobsuffix="km$system"       
elif [ $modelname == "sasha9b200" ]
then
    jobsuffix="kn$system"       
elif [ $modelname == "sasha99" ]
then
    jobsuffix="jz$system"       
elif [ $modelname == "thickdiskr7" ]
then
    jobsuffix="ko$system"
elif [ $modelname == "sashaa99t0.15" ]
then
    jobsuffix="aa$system"
elif [ $modelname == "sashaa99t0.3" ]
then
    jobsuffix="ab$system"
elif [ $modelname == "sashaa99t0.6" ]
then
    jobsuffix="ac$system"
elif [ $modelname == "sashaa99t1.5708" ]
then
    jobsuffix="ad$system"
elif [ $modelname == "sashaa9b100t0.15" ]
then
    jobsuffix="ba$system"
elif [ $modelname == "sashaa9b100t0.3" ]
then
    jobsuffix="bb$system"
elif [ $modelname == "sashaa9b100t0.6" ]
then
    jobsuffix="bc$system"
elif [ $modelname == "sashaa9b100t1.5708" ]
then
    jobsuffix="bd$system"
elif [ $modelname == "sashaam9full2pit0.15" ]
then
    jobsuffix="ca$system"
elif [ $modelname == "sashaam9full2pit0.3" ]
then
    jobsuffix="cb$system"
elif [ $modelname == "sashaam9full2pit0.6" ]
then
    jobsuffix="cc$system"
elif [ $modelname == "sashaam9full2pit1.5708" ]
then
    jobsuffix="cd$system"
elif [ $modelname == "thickdiskfull3d7tilt0.35" ]
then
    jobsuffix="da$system"
elif [ $modelname == "thickdiskfull3d7tilt0.7" ]
then
    jobsuffix="db$system"
elif [ $modelname == "thickdiskfull3d7tilt1.5708" ]
then
    jobsuffix="dc$system"
else
    jobsuffix="$system"        
fi


# defaults
chunklisttypeplot=1
numtasksplot=1
chunklistplot=\"`seq -s " " 1 $numtasksplot`\"
runnplot=1



# runn is number of runs (and in parallel, should be multiple of numcorespernode)


##############################################
# orange
if [ $system -eq 1 ]
then
    # up to 768 cores (96*2*4).  That is, 8cores/node
    # but only 4GB/core.  Seems to work, but maybe much slower than would be if used 6 cores to allow 5.3G/core?
    # ok, now need 5 cores
    # ok, now 3 required for thickdisk7
    # need 2 for thickdisk3 where final qty file is 12GB right now
    numcorespernode=3
    #
    numnodes=$((180/$numcorespernode)) # so 180 cores total
    thequeue="kipac-ibq"
    # first part of name gets truncated, so use suffix instead for reliability no matter how long the names are
fi

##############################################
# orange-gpu
if [ $system -eq 2 ]
then
    numcorespernode=8
    numnodes=3
    # there are 16 cores, but have to use 8 for kipac-gpuq too since only 48GB memory and would use up to 5.3GB/core
    # only 3 free nodes for kipac-gpuq (rather than 4 since one is head node)
    thequeue="kipac-gpuq"
fi

#############################################
# ki-jmck
if [ $system -eq 3 ]
then
    # 4 for thickdisk7 (until new memory put in)
    numcorespernode=1  # MAVARA
    numnodes=1
    thequeue="none"
fi


#############################################
# Nautilus or Kraken (partially)
if [ $system -eq 4 ] ||
    [ $system -eq 5 ] #|| [ $system -eq 7 ]    
then
    # go to directory where "dumps" directory is
    # required for Nautilus, else will change to home directory when job starts
    numnodes=1
    thequeue="standard"
    #
    if [ "$modelname" == "thickdisk7" ] ||
        [ "$modelname" == "thickdiskr7" ] ||
        [ "$modelname" == "thickdiskhr3" ]
    then
        # 24 hours is good enough for these if using 450 files (taking 18 hours for thickdisk7), but not much more.
        timetot="24:00:00"
        numcorespernode=80
        # grep "memoryusage" python*full.out | sed 's/memoryusage=/ /g'|awk '{print $2}' | sort -g
        # thickdisk7 needs at least 11GB/core according to memory usage print outs, so request 12GB
        # sasha99 needs 7GB/core, so give 8GB
        # sasha9b100 needs 4GB/core, so give 6GB
        # runs like sasha5 need 2GB/core, so give 4GB
        # This will increase the number of cores when qsub called, but numcorespernode is really how many tasks.
        memtot=$((16 + $numcorespernode * 14)) # so real number of cores charged will be >3X numcorespernode.
    elif [ "$modelname" == "sasha99" ]
    then
        # takes 3*6281/1578~12 hours for sasha99 movie
        timetot="24:00:00"
        numcorespernode=80
        memtot=$((8 + $numcorespernode * 8))
    elif [ "$modelname" == "sasham9full2pi" ] ||
        [ "$modelname" == "sasha9b100" ]
    then
        timetot="24:00:00"
        numcorespernode=80
        memtot=$((8 + $numcorespernode * 6))
    elif [ "$modelname" == "sasham9" ] ||
        [ "$modelname" == "sasham5" ] ||
        [ "$modelname" == "sasham2" ] ||
        [ "$modelname" == "sasha0" ] ||
        [ "$modelname" == "sasha1" ] ||
        [ "$modelname" == "sasha2" ] ||
        [ "$modelname" == "sasha5" ] ||
        [ "$modelname" == "sasha9b25" ] ||
        [ "$modelname" == "sasha9b50" ] ||
        [ "$modelname" == "sasha9b200" ] ||
        [ "$modelname" == "runlocaldipole3dfiducial" ] ||
        [ "$modelname" == "blandford3d_new" ] ||
        [ "$modelname" == "a0hr07" ]
    then
        timetot="24:00:00"
        numcorespernode=80
        memtot=$((4 + $numcorespernode * 4))
    elif [ "$modelname" == "sashaam9full2pit0.15" ] ||
        [ "$modelname" == "sashaa9b100t0.15" ] ||
        [ "$modelname" == "sashaa99t0.15" ] ||
        [ "$modelname" == "sashaam9full2pit0.3" ] ||
        [ "$modelname" == "sashaa9b100t0.3" ] ||
        [ "$modelname" == "sashaa99t0.3" ] ||
        [ "$modelname" == "sashaam9full2pit0.6" ] ||
        [ "$modelname" == "sashaa9b100t0.6" ] ||
        [ "$modelname" == "sashaa99t0.6" ] ||
        [ "$modelname" == "sashaam9full2pit1.5708" ] ||
        [ "$modelname" == "sashaa9b100t1.5708" ] ||
        [ "$modelname" == "sashaa99t1.5708" ]
    then
        numnodes=6 # overwrite numnodes to 6
        timetot="24:00:00" # if use numnodes=6, only need ~12 hours actually for 5194 images
        numcorespernode=80
        # for new script with reinterp3dspc, uses up to 10GB per core, so give 11GB per core
        memtot=$((11 + $numcorespernode * 11))
    elif [ "$modelname" == "thickdiskfull3d7tilt0.35" ] ||
        [ "$modelname" == "thickdiskfull3d7tilt0.7" ] ||
        [ "$modelname" == "thickdiskfull3d7tilt1.5708" ]
    then
        timetot="24:00:00"
        numcorespernode=80
        # give an extra 2GB/core for these models compared to without reinterp3dspc
        memtot=$((18 + $numcorespernode * 16))
    else
        # default for lower res thick disk poloidal and toroidal runs
        echo "Ended up in default for timetot, numcorespernode, and memtot in makemovie.sh"
        timetot="2:00:00"   #MMMMMMMMMMMMMMMMM
        numcorespernode=80
        memtot=$((4 + $numcorespernode * 4))
    fi
    #
    # for makeplot part or makeplotavg part
    numcorespernodeplot=1
    numnodesplot=1
    # new analysis can take a long time.
    timetotplot="2:00:00" #MMMMMMMMMMMMMMM
    # don't always need so much memory.
    memtotplot=16 #32 MAVARA
    # interactive use for <1hour:
    # ipython -pylab -colors=LightBG
    #
    # long normal interactive job:
    # qsub -I -A TG-PHY120005 -q analysis -l ncpus=8,walltime=24:00:00,mem=32GB
    # Note that this gives you a node for <=24 hours, so you can run 8 processes in parallel.  If you want to open a few xterm's from that window with (xterm &), in that terminal you will have to set the DISPLAY variable to whatever it was in the login node of nautilus (or else, the display variable is empty, and the new xterm windows refuse to spawn).
fi


# Nautilus fix
# this starts a bunch of single node jobs
if [ $system -eq 4 ]
then
    numcorespernodeeff=$(($memtot / 4))
fi

# Kraken
# this starts a bunch of single node jobs
if [ $system -eq 5 ] &&
    [ $parallel -eq 1 ]
then
    thequeue="normal"
    timetot="8:00:00" # probably don't need all this is 1 task per fieldline file

    # Kraken only has 16GB per node of 12 cores
    # so determine how many nodes need based upon Nautilus/Kraken memtot above

    # numtasks set equal to total number of time slices, so each task does only 1 fieldline file
    numtasks=`ls dumps/fieldline*.bin |wc -l`
    memtotnaut=$memtot
    numcorespernodenaut=$numcorespernode

    # total memory required is old memtot/numcorespernode from Nautilus multiplied by total number of tasks for Kraken
    memtotpercore=$((1+$memtotnaut/$numcorespernodenaut))
    
    numcorespernode=$((16/$memtotpercore))

    if [ $numcorespernode -eq 0 ]
    then
        echo "Not enough memory per core to do anything"
        exit
    fi

    numnodes=$(($numtasks/$numcorespernode))
    numnodesreal=$numnodes
    numcorespernodereal=$numcorespernode
    # total number of cores used by these nodes
    #numtotalcores=$(($numnodes*12))
    # numtotalcores is currently what is allocated per node because each job is for each node
    numtotalcores=12

    apcmd="mpiexec -ppn 12 -np 1" #"aprun -n 1 -d 12 -cc none -a xt"

    # setup plotting part
    numnodesplot=1
    numcorespernodeplot=12
    # this gives 16GB free for plotting
    numtotalcoresplot=$numnodesplot
    thequeueplot="normal"
    apcmdplot="mpiexec -ppn 12 -np 1" #"aprun -n 1 -d 12 -cc none -a xt"
    timetotplot="8:00:00"

fi


# Kraken using makemoviec to start fake-MPI job
# this starts a single many-node-core job
# script views this as 1 node with many cores
if [ $system -eq 5 ] &&
    [ $parallel -eq 2 ]
then
    # Kraken only has 16GB per node of 12 cores
    # so determine how many nodes need based upon Nautilus/Kraken memtot above
    memtotnaut=$memtot #324 from above is the default
    numcorespernodenaut=$numcorespernode  #80 is the default

    # numtasks set equal to total number of time slices, so each task does only 1 fieldline file
    numtasks=`ls dumps/fieldline*.bin |wc -l` #2000 or so

    # total memory required is old memtot/numcorespernode from Nautilus multiplied by total number of tasks for Kraken
    memtotpercore=$((1+$memtotnaut/$numcorespernodenaut)) #324/80 ~4 as expected from def of memtot
    numcorespernode=12 #$((32/$memtotpercore))  # was 24. 32 is the total per node for westemere nodes #MAVARA turned to 12 for low res sim since don't need much mem.

    #MAVARA for now just have:
    #numcorespernode=12 #for westemere
    if [ $numcorespernode -eq 0 ]
    then
        echo "Not enough memory per core to do anything"
        exit
    fi

    if [ $numcorespernode -eq 2 ] ||
        [ $numcorespernode -eq 4 ] ||
        [ $numcorespernode -eq 6 ] ||
        [ $numcorespernode -eq 8 ] ||
        [ $numcorespernode -eq 10 ] ||
        [ $numcorespernode -eq 12 ]
    then
        numcorespersocket=$(($numcorespernode/2))
	numtaskshalf=$(($numtasks/2))
        apcmd="mpiexec " #"aprun -n $numtasks -S $numcorespersocket"
    else
        # odd, so can't use socket version
	numtaskshalf=$(($numtasks/2))
        apcmd="mpiexec "#-ppn $numcorespernode -np $numtasks" #mpiexec -np $numtasks -ppn $numcorespernode"
    fi

    numnodes=$(($numtasks/$numcorespernode+1)) #MAVARACHANGE added +1 on may 6th 2014 so actually enough nodes used#MAVARA added /2 since numparts changed to 2 in sections where mpiexec is called with qsub
    numtotalcores=$(($numnodes * 12)) # always 12 for Kraken   # also 12 for westemere nodes on pleiades

    if [ $numtotalcores -le 1024 ]
        then
        thequeue="devel"
    elif [ $numtotalcores -le 8192 ]
        then
        thequeue="long"
    elif [ $numtotalcores -le 49536 ]
        then
        thequeue="large"
    fi

    # GODMARK: 458 thickdisk7 files only took 1:45 on Kraken
    timetot="2:00:00" # probably don't need all this is 1 task per fieldline file #MMMMMMMMMMMMMMM

    echo "PART1: $numcorespernode $numcorespersocket $numnodes $numtotalcores $thequeue $timetot"
    echo "PART2: $apcmd"

    # setup fake setup
    numcorespernodereal=$numcorespernode  #as necessary for enough memory per task on each cpu
    numnodesreal=$numnodes
    numcorespernode=$numtasks
    numnodes=1
    #
    chunklisttype=0
    chunklist=\"`seq -s " " 1 $numtasks`\"
    DATADIR=$dirname
    



    # setup plotting part
    numtasksplot=1
    numnodesplot=1
    numcorespernodeplot=12
    # this gives 16GB free for plotting (temp vars + qty2.npy file has to be smaller than this or swapping will occur)
    numtotalcoresplot=$numcorespernodeplot
    thequeueplot="devel"
    apcmdplot="mpiexec -np $numtasksplot" #mpiexec -np $numtasksplot" #"aprun -n $numtasksplot"
    # only took 6 minutes for thickdisk7 doing 458 files inside qty2.npy!  Up to death at point when tried to resample in time.
    timetotplot="2:00:00" #MMMMMMMMMMMMMMM


    echo "PART1P: $numcorespernodeplot $numnodesplot $numtotalcoresplot $thequeueplot $timetotplot"
    echo "PART2P: $apcmdplot"

fi

#############################################
# Deepthought2 (partially)
if [ $system -eq 7 ]             #### This if-then constains stuff set for and used within both parallel ==1 and ==2 sections below.

then
    # go to directory where "dumps" directory is
    # required for Nautilus, else will change to home directory when job starts
    # init defaults
    numnodes=1
    thequeue="standard"
    #
    # defaults
    echo "Ended up in default for timetot, numcorespernode, and memtot in makemovie.sh"
    timetot="24:00:00"   #MMMMMMMMMMMMMMMMM
    numcorespernode=20
    memtot=$(($numcorespernode * 6)) #actually 6.4 but didn't work with non integer
    
    #
    # for makeplot part or makeplotavg part
    numcorespernodeplot=1 #1
    numnodesplot=1
    # new analysis can take a long time.
    timetotplot="24:00:00" #MMMMMMMMMMMMMMM
    # don't always need so much memory.
    memtotplot=128 #MAVARA
    # interactive use for <1hour:
    # ipython -pylab -colors=LightBG
    #
    # long normal interactive job:
    # qsub -I -A TG-PHY120005 -q analysis -l ncpus=8,walltime=24:00:00,mem=32GB
    # Note that this gives you a node for <=24 hours, so you can run 8 processes in parallel.  If you want to open a few xterm's from that window with (xterm &), in that terminal you will have to set the DISPLAY variable to whatever it was in the login node of nautilus (or else, the display variable is empty, and the new xterm windows refuse to spawn).
fi



# Kraken
# this starts a bunch of single node jobs
if [ $system -eq 7 ] &&
    [ $parallel -eq 1 ]
then
    thequeue="standard"
    timetot="8:00:00" # probably don't need all this is 1 task per fieldline file

    # Kraken only has 16GB per node of 12 cores
    # so determine how many nodes need based upon Nautilus/Kraken memtot above

    # numtasks set equal to total number of time slices, so each task does only 1 fieldline file
    numtasks=`ls dumps/fieldline*.bin |wc -l`
    memtotnaut=$memtot
    numcorespernodenaut=$numcorespernode

    # total memory ***required*** is old memtot/numcorespernode from Nautilus multiplied by total number of tasks for Kraken
    memtotpercore=$((1+$memtotnaut/$numcorespernodenaut))
    
    numcorespernode=$((128/$memtotpercore))

    if [ $numcorespernode -eq 0 ]
    then
        echo "Not enough memory per core to do anything"
        exit
    fi

    numnodes=$((1+$numtasks/$numcorespernode))
    numcorespernodereal=$numcorespernode
    numnodesreal=$numnodes
    # total number of cores used by these nodes
    #numtotalcores=$(($numnodes*12))
    # numtotalcores is currently what is allocated per node because each job is for each node
    numtotalcores=20

    apcmd="mpiexec -n 32" #-ppn 20 -np 1" #"aprun -n 1 -d 12 -cc none -a xt"

    # setup plotting part
    numnodesplot=1
    numcorespernodeplot=20
    # this gives 16GB free for plotting
    numtotalcoresplot=$numnodesplot
    thequeueplot="standard"
    apcmdplot="mpiexec -n 1" #-n $numtasks" # #"aprun -n 1 -d 12 -cc none -a xt"
    timetotplot="8:00:00"

fi


# Kraken using makemoviec to start fake-MPI job
# this starts a single many-node-core job
# script views this as 1 node with many cores
if [ $system -eq 7 ] &&
    [ $parallel -eq 2 ]
then
    # Kraken only has 16GB per node of 12 cores
    # so determine how many nodes need based upon Nautilus/Kraken memtot above
    memtotnaut=$memtot #324 from above is the default
    numcorespernodenaut=$numcorespernode  #80 is the default

    # numtasks set equal to total number of time slices, so each task does only 1 fieldline file
    numtasks=100 #`ls dumps/fieldline*.bin |wc -l` #2000 or so

    # total memory required is old memtot/numcorespernode from Nautilus multiplied by total number of tasks for Kraken
    memtotpercore=$((1+$memtotnaut/$numcorespernodenaut)) #324/80 ~4 as expected from def of memtot
    numcorespernode=20 #$((128/$memtotpercore))  # was 24. 32 is the total per node for westemere nodes #MAVARA turned to 12 for low res sim since don't need much mem.

    #MAVARA for now just have:
    #numcorespernode=12 #for westemere
    if [ $numcorespernode -eq 0 ]
    then
        echo "Not enough memory per core to do anything"
        exit
    fi

    if [ $numcorespernode -eq 2 ] ||
        [ $numcorespernode -eq 4 ] ||
        [ $numcorespernode -eq 6 ] ||
        [ $numcorespernode -eq 8 ] ||
        [ $numcorespernode -eq 10 ] ||
        [ $numcorespernode -eq 12 ] ||
	[ $numcorespernode -eq 20 ]
    then
        numcorespersocket=$(($numcorespernode/2))
	numtaskshalf=$(($numtasks/100 ))
        apcmd="mpirun -ppn $numcorespernode -np $numtasks" # -n $numtasks" #"aprun -n $numtasks -S $numcorespersocket"
    else
        # odd, so can't use socket version
	numtaskshalf=$(($numtasks/100 ))
        apcmd="mpirun -ppn $numcorespernode -np $numtasks" # -n $numtasks" #mpiexec -np $numtasks -ppn $numcorespernode"
    fi

    numnodes=$(($numtasks/$numcorespernode + 1)) #either need to be exactly divisible or add +1 ::: MAVARACHANGE added +1 on may 6th 2014 so actually enough nodes used#MAVARA added /2 since numparts changed to 2 in sections where mpiexec is called with qsub
    numtotalcores=$(($numnodes * 20 )) ##/10 # always 12 for Kraken   # also 12 for westemere nodes on pleiades

    if [ $numtotalcores -le 20 ]
        then
        thequeue="debug"
        thequeueplot="debug"
	timetot="00:15:00" # probably don't need all this is 1 task per fieldline file #MMMMMMMMMMMMMMM
	timetotplot="00:15:00" #MMMMMMMMMMMMMMM    
    elif [ $numtotalcores -le 8192 ]
        then
        thequeue="standard"
	thequeueplot="standard"
	timetot="24:00:00" # probably don't need all this is 1 task per fieldline file #MMMMMMMMMMMMMMM
	timetotplot="24:00:00" #MMMMMMMMMMMMMMM
    elif [ $numtotalcores -le 49536 ]
        then
        thequeue="large"
    fi

    # GODMARK: 458 thickdisk7 files only took 1:45 on Kraken
    

    echo "PART1: $numcorespernode $numcorespersocket $numnodes $numtotalcores $thequeue $timetot"
    echo "PART2: $apcmd"

    # setup fake setup
    numcorespernodereal=$numcorespernode  #as necessary for enough memory per task on each cpu
    numnodesreal=$numnodes
    numcorespernode=$(($numtasks)) ##/10
    numnodes=1
    #
    chunklisttype=1
    chunklist=\"`seq -s " " 1 $numtasks`\" ## was numtasks but changed with numparts = 10
    DATADIR=$dirname
    



    # setup plotting part
    numtasksplot=1
    numnodesplot=1
    numcorespernodeplot=20
    # this gives 16GB free for plotting (temp vars + qty2.npy file has to be smaller than this or swapping will occur)
    numtotalcoresplot=$numcorespernodeplot

    apcmdplot="mpirun -np $numtasksplot" #mpiexec -np $numtasksplot" #"aprun -n $numtasksplot"
    # only took 6 minutes for thickdisk7 doing 458 files inside qty2.npy!  Up to death at point when tried to resample in time.



    echo "PART1P: $numcorespernodeplot $numnodesplot $numtotalcoresplot $thequeueplot $timetotplot"
    echo "PART2P: $apcmdplot"

fi

#############################################




#############################################
# physics-179
if [ $system -eq 6 ]
then
    numcorespernode=16
    numnodes=1
    thequeue="none"
fi




if [ $useoverride -eq 0 ]
then
    runnglobal=$(( $numcorespernode * $numnodes )) ##*10
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
#MAVARA pay attention to above? do I need to run generate_time_series() then?   this is a python function, i think, called through main??

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
#  In order to run mkavgfigs(), you need to have generated 2D
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


###################
#
# some python setup stuff
#
###################
# copy over python script path since supercomputers (e.g. Kraken) can't access home directory while running.
#rm -rf $dirname/py/
echo "dirname=$dirname"
cp -a $HOME/py $dirname/
# setup py path
MYPYTHONPATH=$dirname/py/
MREADPATH=$MYPYTHONPATH/mread/
initfile=$MREADPATH/__init__.py
echo "initfile=$initfile"
myrand=${RANDOM}
echo "RANDOM=$myrand"

# assumes chose correct python library setup and system in Makefile
if [ $parallel -eq 2 ]
then
    # create makemoviec for local use
    oldpath=`pwd`
    cd $MYPYTHONPATH/scripts/

    if [ $system -eq 3 ]
    then
        sed -e 's/USEKIJMCK=0/USEKIJMCK=1/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USEMPI=1/USEMPI=0/g' Makefile > Makefile.temp
        cp Makefile.temp Makefile
    elif [ $system -eq 4 ]
    then
        sed -e 's/USEKIJMCK=1/USEKIJMCK=0/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=0/USENAUTILUS=1/g' -e 's/USEMPI=0/USEMPI=1/g' Makefile > Makefile.temp
        cp Makefile.temp Makefile
    elif [ $system -eq 5 ]
    then
        sed -e 's/USEKIJMCK=1/USEKIJMCK=0/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USEMPI=0/USEMPI=1/g' Makefile > Makefile.temp  #MMMMMAVARA usekraken=0 so usepfe can be 1
        cp Makefile.temp Makefile
    elif [ $system -eq 7 ]
    then
        sed -e 's/USEKIJMCK=1/USEKIJMCK=0/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USEMPI=0/USEMPI=1/g' Makefile > Makefile.temp  #MMMMMAVARA usekraken=0 so usepfe can be 1
        cp Makefile.temp Makefile
    elif [ $system -eq 6 ]
    then
        sed -e 's/USEKIJMCK=0/USEKIJMCK=1/g' -e 's/USEKRAKEN=1/USEKRAKEN=0/g' -e 's/USENAUTILUS=1/USENAUTILUS=0/g' -e 's/USEMPI=1/USEMPI=0/g' Makefile > Makefile.temp
        cp Makefile.temp Makefile
    else
        echo "Not setup for system=$sytem"
        exit
    fi

    make clean ; make
    cd $oldpath
    makemoviecfullfile=$MYPYTHONPATH/scripts/makemoviec
    chmod ug+rx $makemoviecfullfile

fi



localpath=`pwd`

passpart1="a#"
passpart2="hyq#ng9"

#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit  #MAVARA not sure what this does?

#avoid questions that would stall things
alias cp='cp'
alias rm='rm'
alias mv='mv'

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

    #sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile1
    # no longer edit init, just pass args
    runtype=3
    cp $initfile $myinitfile1


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
    # echo "nohup python $myinitfile1 $runtype $modelname $runi $runn &> python_${runi}_${runn}.out &"
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

              # for parallel -ge 1, do every numcorespernode starting with i=0
	            modi=$(($i % $numcorespernode))

	            dowrite=0
	            if [ $modi -eq 0 ]
		    then
		        dowrite=1
		    fi
	            

	            if [ $dowrite -eq 1 ]
		    then
                        # create script to be run
		        thebatch="sh1_python_${i}_${numcorespernode}_${runn}.sh"
		        rm -rf $thebatch
		        echo "j=$j" >> $thebatch
		        echo "itot=$itot" >> $thebatch
		        echo "i=$i" >> $thebatch
		        echo "runn=$runn" >> $thebatch
		        echo "itemspergroup=$itemspergroup" >> $thebatch
		        echo "numcorespernode=$numcorespernode" >> $thebatch
	            fi
	            echo "i=$i numcorespernode=$numcorespernode modi=$modi"
		    
		    if [ $modi -eq 0 ]
		    then
 		        myruni='$(( $cor - 1 + $i + $itot * $j ))' #$i will be an integer multiple of numcorespernode since this is only run when modi eq 0
		        myseq='`seq 1 $numcorespernode`'
		        if [ $dowrite -eq 1 ]
			then
			    echo "for cor in $myseq" >> $thebatch
			    echo "do" >> $thebatch
		        fi
		    fi
	            

	            
	            if [ $dowrite -eq 1 ]
		        then
			echo "cd $dirname" >> $thebatch
			echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $thebatch
			if [ $system -eq 4 ]
			then
                            echo "unset MPLCONFIGDIR" >> $thebatch
			else                        
                            rm -rf $dirname/matplotlibdir/
                            echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $thebatch
			fi
		        echo "runi=$myruni" >> $thebatch
		        echo "textrun=\"Data vs. Time: Running i=\$i j=\$j giving runi=\$runi with runn=\$runn\"" >> $thebatch
		        echo "echo \$textrun" >> $thebatch
		        echo "sleep 1" >> $thebatch
		        cmdraw="python $myinitfile1 $runtype $modelname "'$runi $runn'
		        cmdfull='((nohup $cmdraw 2>&1 1>&3 | tee python_${runi}_${cor}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runi}_${cor}_${runn}.out) > python_${runi}_${cor}_${runn}.full.out 2>&1'
		        echo "cmdraw=\"$cmdraw\"" >> $thebatch
		        echo "cmdfull=\"$cmdfull\"" >> $thebatch
		        echo "echo \"\$cmdfull\" > torun_$thebatch.\$cor.sh" >> $thebatch
		        echo "nohup sh torun_$thebatch.\$cor.sh &" >> $thebatch
	            fi


		    if [ $dowrite -eq 1 ]
		    then
		        echo "done" >> $thebatch
		    fi

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
                            jobcheck=md.$jobsuffix
		            jobname=$jobprefix.${i}.${jobcheck}
		            outputfile=$jobname.out
		            errorfile=$jobname.err
                            rm -rf $outputfile
                            rm -rf $errorfile
                        #
                            if [ $system -eq 4 ]
                            then
		                bsubcommand="qsub1 -S /bin/bash -A TG-PHY120005 -l mem=${memtot}GB,walltime=$timetot,ncpus=$numcorespernodeeff -q $thequeue -N $jobname -o $outputfile -e $errorfile -M jmckinne@stanford.edu ./$thebatch"
                            elif [ $system -eq 7 ]
                            then
				superbatch=superbatchfile.$thebatch
				rm -rf $superbatch
				echo "#!/bin/bash" >> $superbatch
				echo "cd $dirname" >> $superbatch
			        cat ~/setuppython273 >> $superbatch
				# echo "source ~/binliblinks" >> $superbatch
				echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $superbatch
				rm -rf $dirname/matplotlibdir/
				echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $superbatch
		                fakeruni=99999999999999
				if [ $parallel -eq 1 ]
				then
                                    echo "$apcmd ./$thebatch" >> $superbatch
				else
		                    echo "echo $chunklist >> $dirname/chunklistfile.txt" >> $superbatch
				    cmdraw="$makemoviecfullfile $chunklisttype chunklistfile.txt $runn $DATADIR $jobcheck $myinitfile1 $runtype $modelname $fakeruni $runn"
                                    echo "$apcmd $cmdraw" >> $superbatch
				    #echo "mpdallexit" >> $superbatch
				fi
				localerrorfile=python_${fakeruni}_${runn}.stderr.out
				localoutputfile=python_${fakeruni}_${runn}.out
				rm -rf $localerrorfile
				rm -rf $localoutputfile
				sleep 20
				bsubcommand="sbatch -n $numtotalcores -t $timetot -A astronomy -p $thequeue --job-name=$jobname -o $localoutputfile -e $localerrorfile --mail-user=mavara@astro.umd.edu ./$superbatch"
                            else
                            # probably specifying ptile below is not necessary
		                bsubcommand="bsub -n 1 -x -R span[ptile=$numcorespernode] -q $thequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
                            fi
                        #
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
	            
	    done   #for loop over ilist

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
                    if [ $system -eq 4 ] ||
                        [ $system -eq 7 ]
                    then
		        pendjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep " Q " | wc -l`
		        runjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep " R " | wc -l`
                    else
		        pendjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep PEND | wc -l`
		        runjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep RUN | wc -l`
                    fi
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
    
    done # end of for j in numparts
    
    wait

    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile1
    fi

fi


#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


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

    #sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile2
    # no longer edit init, just pass args
    runtype=3
    cp $initfile $myinitfile2


    echo "Merge to single file"
    if [ $testrun -eq 1 ]
    then
	echo "((nohup python $myinitfile2 $runtype $modelname $runn $runn 2>&1 1>&3 | tee python_${runn}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.out) > python_${runn}_${runn}.full.out 2>&1"
    else
	((nohup python $myinitfile2 $runtype $modelname $runn $runn 2>&1 1>&3 | tee python_${runn}_${runn}.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.out) > python_${runn}_${runn}.full.out 2>&1
    fi

    if [ $rminitfiles -eq 1 ]
    then
        # remove created file
	rm -rf $myinitfile2
    fi

fi


#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# Make plots and latex tables
#
####################################
if [ $makeplot -eq 1 ]
then

    myinitfile3=$localpath/__init__.py.3.$myrand
    echo "myinitfile3="${myinitfile3}

    #sed -n '1h;1!H;${;g;s/if False:[\n \t]*#NEW FORMAT[\n \t]*#Plot qtys vs. time[\n \t]*generate_time_series()/if True:\n\t#NEW FORMAT\n\t#Plot qtys vs. time\n\tgenerate_time_series()/g;p;}'  $initfile > $myinitfile3
    runtype=3
    cp $initfile $myinitfile3


    echo "Generate the plots"


    ##########################################################################
    # string "plot" tells script to do plot
    thebatch="sh1_pythonplot_${numcorespernodeplot}.sh"
    rm -rf $thebatch
    echo "cd $dirname" >> $thebatch
    echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $thebatch
    if [ $system -eq 4 ]
    then
        echo "unset MPLCONFIGDIR" >> $thebatch
    else                        
        rm -rf $dirname/matplotlibdir/
        echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $thebatch
    fi
    echo "((nohup python $myinitfile3 $runtype $modelname plot $makepowervsmplots $makespacetimeplots $makefftplot $makespecplot $makeinitfinalplot $makethradfinalplot 2>&1 1>&3 | tee python.plot.stderr.out) 3>&1 1>&2 | tee python.plot.out) > python.plot.full.out 2>&1" >> $thebatch
    echo "wait" >> $thebatch
    chmod a+x $thebatch

    dowrite=1
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
            jobcheck=pl.$jobsuffix
	    jobname=$jobprefix.${jobcheck}
	    outputfile=$jobname.pl.out
	    errorfile=$jobname.pl.err
            rm -rf $outputfile
            rm -rf $errorfile
            #
            if [ $system -eq 4 ]
            then
		bsubcommand="qsub2 -S /bin/bash -A TG-PHY120005 -l mem=${memtotplot}GB,walltime=$timetotplot,ncpus=$numcorespernodeplot -q $thequeue -N $jobname -o $outputfile -e $errorfile ./$thebatch"
            elif [ $system -eq 7 ]
            then
                superbatch=superbatchfile.$thebatch
                rm -rf $superbatch
                echo "#!/bin/bash" >> $superbatch
		echo "cd $dirname" >> $superbatch
                cat ~/setuppython273 >> $superbatch
		#echo "source ~/binliblinks" >> $superbatch
                rm -rf $dirname/matplotlibdir/
                echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $superbatch
                echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $superbatch
                if [ $parallel -eq 1 ]
                then
                    echo "$apcmdplot ./$thebatch" >> $superbatch
                else
		    echo "echo $chunklistplot >> $dirname/chunklistfile2.txt" >> $superbatch
		    cmdraw="$makemoviecfullfile $chunklisttypeplot $dirname/chunklistfile2.txt $runnplot $DATADIR $jobcheck $myinitfile3 $runtype $modelname plot $makepowervsmplots $makespacetimeplots $makefftplot $makespecplot $makeinitfinalplot $makethradfinalplot"
                    echo "$apcmdplot $cmdraw" >> $superbatch
		    #echo "mpdallexit" >> $superbatch
                fi
                localerrorfile=python.plot.stderr.out
                localoutputfile=python.plot.out
                rm -rf $localerrorfile
                rm -rf $localoutputfile
		sleep 20
                #
		bsubcommand="sbatch -N $numnodesplot --ntasks-per-node=20 -t $timetotplot -A astronomy -p $thequeueplot --job-name=$jobname -o $localoutputfile -e $localerrorfile --mail-user=mavara@astro.umd.edu ./$superbatch"
            else
                # probably specifying ptile below is not necessary
		bsubcommand="bsub -n 1 -x -R span[ptile=$numcorespernodeplot] -q $thequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
            fi
                #
	    if [ $testrun -eq 1 ]
	    then
		echo $bsubcommand
	    else
		echo $bsubcommand
		echo "$bsubcommand" > bsubshtorun_pl_$thebatch
		chmod a+x bsubshtorun_pl_$thebatch
		sh bsubshtorun_pl_$thebatch
	    fi
	fi
    fi



    ##########################################################################
    # waiting game
    if [ $parallel -eq 0 ]
    then
	wait
    else
                # Wait to move on until all jobs done
	totaljobs=1
	firsttimejobscheck=1
	while [ $totaljobs -gt 0 ]
	do
            if [ $system -eq 4 ] ||
                [ $system -eq 7 ]
            then
		pendjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep mavara | grep " Q " | wc -l`
		runjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep mavara | grep " R " | wc -l`
            else
		pendjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep PEND | wc -l`
		runjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep RUN | wc -l`
            fi
	    totaljobs=$(($pendjobs+$runjobs))
	    
	    if [ $totaljobs -gt 0 ]
	    then
		echo "PEND=$pendjobs RUN=$runjobs TOTAL=$totaljobs ... waiting ..."
		sleep 10
                firsttimejobscheck=0
	    else
		if [ $firsttimejobscheck -eq 1 ]
		then
		    totaljobs=1
		    echo "waiting for jobs to get started..."
		    sleep 10
		else
		    echo "DONE!"		      
		fi
	    fi
	done
        
    fi
    
    wait
    ##########################################################################



    
    #########################################################
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



#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


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
    
    #sed -n '1h;1!H;${;g;s/if False:[\n \t]*#make a movie[\n \t]*mkmovie()/if True:\n\t#make a movie\n\tmkmovie()/g;p;}'  $initfile > $myinitfile4
    runtype=4    # MAVARA This is set here, read into main() in __init__.py as an argument that then sets the run type and the functions in __init__ that are called.
    cp $initfile $myinitfile4
    
    
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

    resid=$(( $runn - $itot * $numparts ))
    
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

              # for parallel -ge 1, do every numcorespernode starting with i=0
	        modi=$(($i % $numcorespernode))

	        dowrite=0
		if [ $modi -eq 0 ]
		then
		    dowrite=1
		fi

	        if [ $dowrite -eq 1 ]
		then
                  # create script to be run
		    thebatch="sh4_python_${i}_${numcorespernode}_${runn}.sh"
		    rm -rf $thebatch
		    echo "j=$j" >> $thebatch
		    echo "itot=$itot" >> $thebatch
		    echo "i=$i" >> $thebatch
		    echo "runn=$runn" >> $thebatch
		    echo "itemspergroup=$itemspergroup" >> $thebatch
		    echo "numcorespernode=$numcorespernode" >> $thebatch
	        fi
	        
		echo "i=$i numcorespernode=$numcorespernode modi=$modi"
		if [ $modi -eq 0 ]
		then
 		    myruni='$(( $cor - 1 + $i + $itot * $j ))'
		    myseq='`seq 1 $numcorespernode`'
		    if [ $dowrite -eq 1 ]
		    then
			echo "for cor in $myseq" >> $thebatch
			echo "do" >> $thebatch
		    fi
		fi

	        
	        if [ $dowrite -eq 1 ]
		then
                    echo "cd $dirname" >> $thebatch
                    echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $thebatch
                    if [ $system -eq 4 ]
                    then
                        echo "unset MPLCONFIGDIR" >> $thebatch
                    else                        
                        rm -rf $dirname/matplotlibdir/
                        echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $thebatch
                    fi
		    echo "runi=$myruni" >> $thebatch
		    echo "textrun=\"Movie Frames vs. Time: Running i=\$i j=\$j giving runi=\$runi with runn=\$runn\"" >> $thebatch
		    echo "echo \$textrun" >> $thebatch
		    echo "sleep 1" >> $thebatch
		    cmdraw="python $myinitfile4 $runtype $modelname "'$runi $runn'
		    cmdfull='((nohup $cmdraw 2>&1 1>&3 | tee python_${runi}_${cor}_${runn}.stderr.movieframes.out) 3>&1 1>&2 | tee python_${runi}_${cor}_${runn}.movieframes.out) > python_${runi}_${cor}_${runn}.full.movieframes.out 2>&1'
		    echo "cmdraw=\"$cmdraw\"" >> $thebatch
		    echo "cmdfull=\"$cmdfull\"" >> $thebatch
		    echo "echo \"\$cmdfull\" > torun_$thebatch.\$cor.sh" >> $thebatch
		    echo "nohup sh torun_$thebatch.\$cor.sh &" >> $thebatch
	        fi


	       	if [ $dowrite -eq 1 ]
		then
		    echo "done" >> $thebatch
		fi
	        
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
                        jobcheck=mv.$jobsuffix
		        jobname=$jobprefix.${i}.${jobcheck}
		        outputfile=$jobname.out
		        errorfile=$jobname.err
                        rm -rf $outputfile
                        rm -rf $errorfile
                        #
                        if [ $system -eq 4 ]
                        then
		            bsubcommand="qsub3 -S /bin/bash -A TG-PHY120005 -l mem=${memtot}GB,walltime=$timetot,ncpus=$numcorespernodeeff -q $thequeue -N $jobname -o $outputfile -e $errorfile ./$thebatch"
                        elif [ $system -eq 7 ]
                        then
                            superbatch=superbatchfile.$thebatch
                            rm -rf $superbatch
			    echo "#!/bin/bash" >> $superbatch
                            echo "cd $dirname" >> $superbatch
                            cat ~/setuppython273 >> $superbatch
			    #echo  "source ~/binliblinks" >> $superbatch
                            rm -rf $dirname/matplotlibdir/
                            echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $superbatch
                            echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $superbatch
		            fakeruni=99999999999999
                            if [ $parallel -eq 1 ]
                            then
                                echo "$apcmd ./$thebatch" >> $superbatch
                            else
				echo "echo $chunklist >> $dirname/chunklistfile3.txt" >> $superbatch
		                cmdraw="$makemoviecfullfile $chunklisttype $dirname/chunklistfile3.txt $runn $DATADIR $jobcheck $myinitfile4 $runtype $modelname $fakeruni $runn"
                                echo "$apcmd $cmdraw" >> $superbatch
				#echo "mpdallexit" >> $superbatch
                            fi
		            localerrorfile=python_${fakeruni}_${runn}.stderr.movieframes.out
                            localoutputfile=python_${fakeruni}_${runn}.movieframes.out
                            rm -rf $localerrorfile
                            rm -rf $localoutputfile
			    sleep 20
                            #
		            bsubcommand="sbatch -n $numtotalcores -t $timetot -A astronomy -p $thequeue --job-name=$jobname -o $localoutputfile -e $localerrorfile --mail-user=mavara@astro.umd.edu ./$superbatch"
                        else
                            # probably specifying ptile below is not necessary
		            bsubcommand="bsub -n 1 -x -R span[ptile=$numcorespernode] -q $thequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
                        fi
                        #
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
                    if [ $system -eq 4 ] ||
                        [ $system -eq 7 ]
                    then
		        pendjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep mavara | grep " Q " | wc -l`
		        runjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep mavara | grep " R " | wc -l`
                    else
		        pendjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep PEND | wc -l`
		        runjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep RUN | wc -l`
                    fi
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

#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


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


#	        ffmpeg -y -fflags +genpts -r $fps -i initfinal%04d.png -vcodec mpeg4 -sameq -qmax 5 initfinal.$modelname.avi
#	        ffmpeg -y -fflags +genpts -r $fps -i stream%04d.png -vcodec mpeg4 -sameq -qmax 5 stream.$modelname.avi



    echo "Now do: mplayer -loop 0 lrho.$modelname.avi OR lrhosmall.$modelname.avi"

fi


#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# avg File (takes average of $itemspergroup fieldline files per avg file created)
#
####################################

#<<<<<<< HEAD
#itemspergroup=$(( 6 )) # MAVARA
##itemspergroup=$(( 20 ))
#=======
numfiles=`find dumps/ -name "fieldline*.bin"|wc -l`

echo "NUMFILES=$numfiles"

#itemspergroup=$(( 1 )) # MAVARA
itemspergroup=$(( 1 )) # 16 with 100

# catch too small number of files
# must match __init__.py
if [ $numfiles -le $itemspergroup ]
then
    if [ $numfiles -eq 1 ]
    then
        itemspergroup=1
    else
        itemspergroup=$(( $numfiles - 1))
    fi
fi

#>>>>>>> jon
itemspergrouptext=`printf "%02d"  "$itemspergroup"`


if [ $makeavg -eq 1 ]
then

    echo "Doing avg file"

    runn=${runnglobal} #num field line files 41
    numparts=1


    myinitfile5=$localpath/__init__.py.5.$myrand
    echo "myinitfile5="${myinitfile5}

    #sed -n '1h;1!H;${;g;s/if False:[\n \t]*#2DAVG[\n \t]*mk2davg()/if True:\n\t#2DAVG\n\tmk2davg()/g;p;}'  $initfile > $myinitfile5
    runtype=2
    cp $initfile $myinitfile5

    
    je=$(( $numparts - 1 )) # 0
    # above two must be exactly divisible
    itot=$(( $runn/$numparts )) # 41
    ie=$(( $itot -1 )) #40

    resid=$(( $runn - $itot * $numparts )) # 0
    
    echo "Running with $itot cores simultaneously"
    
    # just a test:
    # echo "nohup python $myinitfile1 $runtype $modelname $runi $runn &> python_${runi}_${runn}.out &"
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

              # for parallel -ge 1, do every numcorespernode starting with i=0
	        modi=$(($i % $numcorespernode))

	        dowrite=0
		if [ $modi -eq 0 ]
		then
		    dowrite=1
		fi
	        

	        if [ $dowrite -eq 1 ] #only happens at first core on each node
		then
                  # create script to be run
		    thebatch="sh5_python_${i}_${numcorespernode}_${runn}.sh"
		    rm -rf $thebatch
		    echo "j=$j" >> $thebatch
		    echo "itot=$itot" >> $thebatch
		    echo "i=$i" >> $thebatch
		    echo "runn=$runn" >> $thebatch
		    echo "itemspergroup=$itemspergroup" >> $thebatch
		    echo "numcorespernode=$numcorespernode" >> $thebatch
	        fi
	        
		echo "i=$i numcorespernode=$numcorespernode modi=$modi"
		if [ $modi -eq 0 ]
		then
 		    myruni='$(( $cor - 1 + $i + $itot * $j ))'
		    myseq='`seq 1 $numcorespernode`'
		    if [ $dowrite -eq 1 ]
		    then
			echo "for cor in $myseq" >> $thebatch
			echo "do" >> $thebatch
		    fi
		fi
	            #fi

	        
	        if [ $dowrite -eq 1 ]
		then
                    echo "cd $dirname" >> $thebatch
                    echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $thebatch
                    if [ $system -eq 4 ]
                    then
                        echo "unset MPLCONFIGDIR" >> $thebatch
                    else                        
                        rm -rf $dirname/matplotlibdir/
                        echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $thebatch
                    fi
		    echo "runi=$myruni" >> $thebatch
		    echo "textrun=\"Avg Data vs. Time: Running i=\$i j=\$j giving runi=\$runi with runn=\$runn\"" >> $thebatch
		    echo "echo \$textrun" >> $thebatch
		    echo "sleep 1" >> $thebatch
		    cmdraw="python $myinitfile5 $runtype $modelname "'$runi $runn $itemspergroup'
		    cmdfull='((nohup $cmdraw 2>&1 1>&3 | tee python_${runi}_${cor}_${runn}.stderr.avg.out) 3>&1 1>&2 | tee python_${runi}_${cor}_${runn}.avg.out) > python_${runi}_${cor}_${runn}.full.avg.out 2>&1'
		    echo "cmdraw=\"$cmdraw\"" >> $thebatch
		    echo "cmdfull=\"$cmdfull\"" >> $thebatch
		    echo "echo \"\$cmdfull\" > torun_$thebatch.\$cor.sh" >> $thebatch
		    echo "sh torun_$thebatch.\$cor.sh &" >> $thebatch
	        fi


	            #if [ $parallel -ge 1 ]
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
                        jobcheck=ma.$jobsuffix
		        jobname=$jobprefix.${i}.${jobcheck}
		        outputfile=$jobname.out
		        errorfile=$jobname.err
                        rm -rf $outputfile
                        rm -rf $errorfile
                        #
                        if [ $system -eq 4 ]
                        then
		            bsubcommand="qsub4 -S /bin/bash -A TG-PHY120005 -l mem=${memtot}GB,walltime=$timetot,ncpus=$numcorespernodeeff -q $thequeue -N $jobname -o $outputfile -e $errorfile ./$thebatch"
                        elif [ $system -eq 7 ]
                        then
                            superbatch=superbatchfile.$thebatch
                            rm -rf $superbatch
			    echo "#!/bin/bash" >> $superbatch
                            echo "cd $dirname" >> $superbatch
                            cat ~/setuppython273 >> $superbatch
			    #echo "source ~/binliblinks" >> $superbatch
                            rm -rf $dirname/matplotlibdir/
                            echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $superbatch
                            echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $superbatch
		            fakeruni=99999999999999
                            if [ $parallel -eq 1 ]
                            then
                                echo "$apcmd ./$thebatch" >> $superbatch
                            else
				echo "echo $chunklist >> $dirname/chunklistfile4.txt" >> $superbatch
		                cmdraw="$makemoviecfullfile $chunklisttype $dirname/chunklistfile4.txt $runn $DATADIR $jobcheck $myinitfile5 $runtype $modelname $fakeruni $runn $itemspergroup"
                                echo "$apcmd $cmdraw" >> $superbatch
				#echo "mpdallexit" >> $superbatch
                            fi
		            localerrorfile=python_${fakeruni}_${runn}.stderr.avg.out
                            localoutputfile=python_${fakeruni}_${runn}.avg.out
                            rm -rf $localerrorfile
                            rm -rf $localoutputfile
			    sleep 20
                            #
		            bsubcommand="sbatch -n $numtotalcores -t $timetot -A astronomy -p $thequeue --job-name=$jobname -o $localoutputfile -e $localerrorfile --mail-user=mavara@astro.umd.edu ./$superbatch"
                        else
                            # probably specifying ptile below is not necessary
		            bsubcommand="bsub -n 1 -x -R span[ptile=$numcorespernode] -q $thequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
                        fi
                        #
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
                    if [ $system -eq 4 ] ||
                        [ $system -eq 7 ]
                    then
		        pendjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep mavara | grep " Q " | wc -l`
		        runjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep mavara | grep " R " | wc -l`
                    else
		        pendjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep PEND | wc -l`
		        runjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep RUN | wc -l`
                    fi
                    #
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

#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


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

    #sed -n '1h;1!H;${;g;s/if False:[\n \t]*#2DAVG[\n \t]*mk2davg()/if True:\n\t#2DAVG\n\tmk2davg()/g;p;}'  $initfile > $myinitfile6
    runtype=2
    cp $initfile $myinitfile6


    echo "Merge avg files to single avg file"
    # <index of first avg file to use> <index of last avg file to use> <step=1>
    # must be step=1, or no merge occurs
    step=1
    whichgroups=$(( 0 ))
    numavg2dmerge=`ls -vrt | egrep "avg2d"${itemspergrouptext}"_[0-9]*\.npy"|wc|awk '{print $1}'`
    #whichgroupe=$(( $itemspergroup * $runn ))
    whichgroupe=$numavg2dmerge

    groupsnum=`printf "%04d" "$whichgroups"`
    groupenum=`printf "%04d" "$whichgroupe"`

    echo "GROUPINFO: $step $itemspergroup $whichgroups $whichgroupe $groupsnum $groupenum"

    if [ $testrun -eq 1 ]
	then
	    echo "((nohup python $myinitfile6 $runtype $modelname $whichgroups $whichgroupe $step 2>&1 1>&3 | tee python_${runn}_${runn}.avg.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.avg.out) > python_${runn}_${runn}.avg.full.out 2>&1"
    else
	    ((nohup python $myinitfile6 $runtype $modelname $whichgroups $whichgroupe $step $itemspergroup 2>&1 1>&3 | tee python_${runn}_${runn}.avg.stderr.out) 3>&1 1>&2 | tee python_${runn}_${runn}.avg.out) > python_${runn}_${runn}.avg.full.out 2>&1

        # copy resulting avg file to avg2d.npy
	    avg2dmerge=`ls -vrt avg2d${itemspergrouptext}_${groupsnum}_${groupenum}.npy | head -1`    
	    cp $avg2dmerge avg2d.npy

    fi

    if [ $rminitfiles -eq 1 ]
	then
        # remove created file
	    rm -rf $myinitfile6
    fi




fi

#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


###################################
#
# Make avg plot
#
####################################
if [ $makeavgplot -eq 1 ]
then

    myinitfile7=$localpath/__init__.py.7.$myrand
    echo "myinitfile7="${myinitfile7}

    #sed -n '1h;1!H;${;g;s/if False:[\n \t]*#fig2 with grayscalestreamlines and red field lines[\n \t]*mkavgfigs()/if True:\n\t#fig2 with grayscalestreamlines and red field lines\n\tmkavgfigs()/g;p;}'  $initfile > $myinitfile7
    runtype=5
    cp $initfile $myinitfile7


    echo "Generate the avg plots"



    




    # string "plot" tells script to do plot
    thebatch="sh1_pythonplot.avg_${numcorespernodeplot}.sh"
    rm -rf $thebatch
    echo "cd $dirname" >> $thebatch
    echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $thebatch
    if [ $system -eq 4 ]
    then
        echo "unset MPLCONFIGDIR" >> $thebatch
    else                        
        rm -rf $dirname/matplotlibdir/
        echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $thebatch
    fi
    echo "((nohup python $myinitfile7 $runtype $modelname 2>&1 1>&3 | tee python.plot.avg.stderr.out) 3>&1 1>&2 | tee python.plot.avg.out) > python.plot.avg.full.out 2>&1" >> $thebatch
    echo "wait" >> $thebatch
    chmod a+x $thebatch


    dowrite=1
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
            jobcheck=pa.$jobsuffix
	    jobname=$jobprefix.${jobcheck}
	    outputfile=$jobname.pa.out
	    errorfile=$jobname.pa.err
            rm -rf $outputfile
            rm -rf $errorfile
            #
            if [ $system -eq 4 ]
            then
		bsubcommand="qsub5 -S /bin/bash -A TG-PHY120005 -l mem=${memtotplot}GB,walltime=$timetotplot,ncpus=$numcorespernodeplot -q $thequeue -N $jobname -o $outputfile -e $errorfile ./$thebatch"
            elif [ $system -eq 7 ]
            then
                superbatch=superbatchfile.$thebatch
                rm -rf $superbatch
		echo "#!/bin/bash" >> $superbatch
                echo "cd $dirname" >> $superbatch
                cat ~/setuppython273 >> $superbatch
		#echo "source ~/binliblinks" >> $superbatch
                rm -rf $dirname/matplotlibdir/
                echo "export MPLCONFIGDIR=$dirname/matplotlibdir/" >> $superbatch
                echo "export PYTHONPATH=$dirname/py:$PYTHONPATH" >> $superbatch
                if [ $parallel -eq 1 ]
                then
                    echo "$apcmdplot ./$thebatch" >> $superbatch
                else
		    echo "echo $chunklistplot >> $dirname/chunklistfile5.txt" >> $superbatch
		    cmdraw="$makemoviecfullfile $chunklisttypeplot $dirname/chunklistfile5.txt $runnplot $DATADIR $jobcheck $myinitfile7 $runtype $modelname"
                    echo "$apcmdplot $cmdraw" >> $superbatch
		    #echo "mpdallexit" >> $superbatch
                fi
                localerrorfile=python.plot.avg.stderr.out
                localoutputfile=python.plot.avg.out
                rm -rf $localerrorfile
                rm -rf $localoutputfile
		sleep 20
                # 
		bsubcommand="sbatch -N $numnodesplot --ntasks-per-node=20 -t $timetotplot -A astronomy -p $thequeueplot --job-name=$jobname -o $localoutputfile -e $localerrorfile --mail-user=mavara@astro.umd.edu ./$superbatch"
            else
                    # probably specifying ptile below is not necessary
		bsubcommand="bsub -n 1 -x -R span[ptile=$numcorespernodeplot] -q $thequeue -J $jobname -o $outputfile -e $errorfile ./$thebatch"
            fi
                #
	    if [ $testrun -eq 1 ]
	    then
		echo $bsubcommand
	    else
		echo $bsubcommand
		echo "$bsubcommand" > bsubshtorun_pl.avg_$thebatch
		chmod a+x bsubshtorun_pl.avg_$thebatch
		sh bsubshtorun_pl.avg_$thebatch
	    fi
	fi
    fi
    

    ##########################################################################
    # waiting game
    if [ $parallel -eq 0 ]
    then
	wait
    else
                # Wait to move on until all jobs done
	totaljobs=1
	firsttimejobscheck=1
	while [ $totaljobs -gt 0 ]
	do
            if [ $system -eq 4 ] ||
                [ $system -eq 7 ]
            then
		pendjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep mavara | grep " Q " | wc -l`
		runjobs=`squeue -p $thequeue -u mavara -o "%.7i %.9P %.80j %.8u %.2t %.9M %.6D %R" 2>&1 | grep $jobcheck | grep mavara | grep " R " | wc -l`
            else
		pendjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep PEND | wc -l`
		runjobs=`bjobs -u all -q $thequeue 2>&1 | grep $jobcheck | grep jmckinn | grep RUN | wc -l`
            fi
	    totaljobs=$(($pendjobs+$runjobs))
	    
	    if [ $totaljobs -gt 0 ]
	    then
		echo "PEND=$pendjobs RUN=$runjobs TOTAL=$totaljobs ... waiting ..."
		sleep 10
                firsttimejobscheck=0
	    else
		if [ $firsttimejobscheck -eq 1 ]
		then
		    totaljobs=1
		    echo "waiting for jobs to get started..."
		    sleep 10
		else
		    echo "DONE!"		      
		fi
	    fi
	done
        
    fi
    
    wait
    ##########################################################################
    


    if [ $rminitfiles -eq 1 ]
    then
        # remove created file
	rm -rf $myinitfile7
    fi

fi


#echo $passpart1$passpart2 | /usr/kerberos/bin/kinit


# to clean-up bad start, use:
# rm -rf sh*.sh bsub*.sh __init* py*.out torun*.sh j1*.err j1*.out *.npy
#
