dirrunstilt="sashaam9full2pit0.15 sashaa9b100t0.15 sashaa99t0.15 sashaam9full2pit0.3 sashaa9b100t0.3 sashaa99t0.3 sashaam9full2pit0.6 sashaa9b100t0.6 sashaa99t0.6 sashaam9full2pit1.5708 sashaa9b100t1.5708 sashaa99t1.5708 thickdiskfull3d7tilt0.35 thickdiskfull3d7tilt0.7 thickdiskfull3d7tilt1.5708"

cd /lustre/medusa/jmckinne/data3/jmckinne/jmckinne/

# fix thickdiskfull3d7 to have tile name
if [ 1 -eq 0 ]
then
    mv thickdiskfull3d7a thickdiskfull3d7tilt0.35a
    mv thickdiskfull3d7b thickdiskfull3d7tilt0.35b
    mv thickdiskfull3d7c thickdiskfull3d7tilt0.35c
    mv thickdiskfull3d7d thickdiskfull3d7tilt0.35d
    mv thickdiskfull3d7e thickdiskfull3d7tilt0.35e
    mv thickdiskfull3d7f thickdiskfull3d7tilt0.35f
    mv thickdiskfull3d7g thickdiskfull3d7tilt0.35g
    mv thickdiskfull3d7h thickdiskfull3d7tilt0.35h
    mv thickdiskfull3d7i thickdiskfull3d7tilt0.35i
    mv thickdiskfull3d7j thickdiskfull3d7tilt0.35j
    mv thickdiskfull3d7k thickdiskfull3d7tilt0.35k
    mv thickdiskfull3d7k.old1 thickdiskfull3d7tilt0.35k.old1
    mv thickdiskfull3d7k.old2 thickdiskfull3d7tilt0.35k.old2
    mv thickdiskfull3d7l thickdiskfull3d7tilt0.35l
    mv thickdiskfull3d7m thickdiskfull3d7tilt0.35m
    mv thickdiskfull3d7n thickdiskfull3d7tilt0.35n
    mv thickdiskfull3d7o thickdiskfull3d7tilt0.35o
    mv thickdiskfull3d7p thickdiskfull3d7tilt0.35p
    mv thickdiskfull3d7q thickdiskfull3d7tilt0.35q
    mv thickdiskfull3d7r thickdiskfull3d7tilt0.35r
    mv thickdiskfull3d7s thickdiskfull3d7tilt0.35s
    mv thickdiskfull3d7t thickdiskfull3d7tilt0.35t
fi

############
for run in $dirrunstilt
do

    cd /lustre/medusa/jmckinne/data3/jmckinne/jmckinne/

    
    sh createlinkssimple.sh $run
    # GDOMARK: have to manually remove bad files!  DO that first, then run full script

    # change to dumps dir
    cd $run/dumps/

    # get last file name/number
    alias ls='ls' # avoid stupid @ in name
    lastrealfile=`ls fieldline*.bin | tail -1`
    
    
    #######
    thedir=$run
    #######
    if [ "$thedir" == "sashaam9full2pit0.15" ] ||
        [ "$thedir" == "sashaam9full2pit0.3" ] ||
        [ "$thedir" == "sashaam9full2pit0.6" ] ||
        [ "$thedir" == "sashaam9full2pit1.5708" ]
    then
        dumpdirnontilt=/lustre/medusa/jmckinne/data1/jmckinne/jmckinne/sasham9full2pi/dumps/
    fi
    #######
    if [ "$thedir" == "sashaa9b100t0.15" ] ||
        [ "$thedir" == "sashaa9b100t0.3" ] ||
        [ "$thedir" == "sashaa9b100t0.6" ] ||
        [ "$thedir" == "sashaa9b100t1.5708" ]
    then
        dumpdirnontilt=/lustre/medusa/jmckinne/data1/jmckinne/jmckinne/sasha9b100/dumps/
    fi
    #######
    if [ "$thedir" == "sashaa99t0.15" ] ||
        [ "$thedir" == "sashaa99t0.3" ] ||
        [ "$thedir" == "sashaa99t0.6" ] ||
        [ "$thedir" == "sashaa99t1.5708" ]
    then
        dumpdirnontilt=/lustre/medusa/jmckinne/data1/jmckinne/jmckinne/sasha99/dumps/
    fi
    #######
    if [ "$thedir" == "thickdiskfull3d7" ] ||
        [ "$thedir" == "thickdiskfull3d7tilt0.7" ] ||
        [ "$thedir" == "thickdiskfull3d7tilt1.5708" ]
    then
        dumpdirnontilt=/lustre/medusa/jmckinne/data1/jmckinne/jmckinne/thickdisk7/dumps/
    fi

    #######
    # put THETAROT0 grid dump file there as required
    ln -s $dumpdirnontilt/gdump.bin gdump.THETAROT0.bin
    #
    # also put fieldline files there (those that don't duplicate existing ones and only before existing starts)
    # but don't overwrite existing links
    ln -s $dumpdirnontilt/fieldline*.bin .
    #
    #
    # now get rid of files after last original file

    ls fieldline*.bin > listfieldline.txt
    linenumberlastgood=`grep -n $lastrealfile listfieldline.txt | sed 's/:/ /g' | awk '{print $1}'`
    linenumberfirstbad=$(( linenumberlastgood + 1 ))
    tail -n +$linenumberfirstbad listfieldline.txt > listfieldlinetoremove.txt
    echo "About to remove files for $run"
    echo "First file is:"
    head -1 listfieldlinetoremove.txt
    for fil in `cat listfieldlinetoremove.txt`
    do
        rm -rf $fil
    done

done