#!/bin/bash
# MUST RUN THIS WITH "bash" not "sh" since on some systems that calls "dash" that doesn't correctly handle $RANDOM or other things



EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` {dirname without trailing /}"
    echo "e.g. sh createlinkes.sh thickdiskrr2"
    exit $E_BADARGS
fi

thedir=$1


# On ki-jmck:

# 1) Ensure sizes all correctly large
#ls -alRS thickdisk11* | grep fieldline | grep -v "\->" | sort -nrk 5

# 2) If not all large, then order list of dirs in order of simulation so overwrites use newer file and repeat

# 3) setup directory list

echo "1"
rm -rf dirs${thedir}.txt
alias ls='ls'
alias lssdir='ls -ap | grep / | sed "s/\///"'
listO=`lssdir  | grep ${thedir}`
list=`echo $listO | sed 's/'${thedir}'\///g' | sed 's/'${thedir}' //g'`

listOnum=`echo $listO | wc -w`
listnum=`echo $list | wc -w`

if [ $listOnum -eq $listnum ]
then
    echo "listOnum and listnum are same!  Didn't remove ${thedir}"
    echo "begin echo of listO"
    echo $listO
    echo "begin echo of list"
    echo $list
    echo "${thedir}" >> /data2/jmckinne/badguys1.txt
    exit
fi

#list=`echo $listO | sed 's/'${thedir}'\///g' | sed 's/'${thedir}' //g' | sed 's/' ${thedir}'//g'`
#
echo "begin echo of list"
echo $list
echo "end echo of list"
for fil in $list ; do echo $fil >> dirs${thedir}.txt ; done

#ls -alrt dirs${thedir}.txt

if [ -e "dirs${thedir}.txt" ]
then
    echo "Got some subparts in dirs${thedir}.txt"
else
    echo "Got no subparts in dirs${thedir}.txt"
    echo "${thedir}" >> /data2/jmckinne/badguys2.txt
    exit
fi

# check that not killing original directory with actual dump data
dirparts=`cat dirs${thedir}.txt`
numdirparts=`echo $dirparts | wc -w`
echo "Number of parts for ${thedir} is $numdirparts"

if [ $numdirparts -lt 1 ]
then
    echo "No sub parts, assume mistake and has original dumps"
    echo "${thedir}" >> /data2/jmckinne/badguys3.txt
    exit
fi


echo "2"
mkdir -p /data2/jmckinne/${thedir}/dumps/
cd /data2/jmckinne/${thedir}/
mv ../dirs${thedir}.txt .

# 4) Edit dir list and choose one's want



# 5) create new full-sim dir and change to dumps dir

echo "3"
cd /data2/jmckinne/${thedir}/dumps/
rm -rf fieldline*.bin
rm -rf dump0000.bin
rm -rf gdump.bin

#exit

# 6) make links
echo "4"
sleep 1


for mydir in `cat ../dirs${thedir}.txt`
do

    # avoid using directory itself in case didn't remove base directory
    if [ "${mydir}" == "${thedir}" ]
    then
        continue
    fi

    echo $mydir
    for fil in `ls ../../$mydir/dumps/fieldline*.bin` 
    do
        echo $fil 
        ln -sf $fil .
    done


    # 7) Also make links to gdump.bin and dump0000.bin
    #firstdir=`head -1 ../dirs${thedir}.txt`
    if [ -e ../../$mydir/dumps/gdump.bin ]
    then
        ln -s ../../$mydir/dumps/gdump.bin .
        ln -s ../../$mydir/dumps/dump0000.bin .
    fi
done


cd ..

echo " "
echo $list
echo "Number of parts for ${thedir} is $numdirparts"

#cp -a ../thickdisk9/movie1 .
#cd movie1/
#rm -rf *.png *.eps *.npy python*.out *.avi *.pdf
#alias cp='cp'
#cp ~/py/scripts/createlinksalt.sh .
#sh createlinksalt.sh 100 ../ ./

# now cd ${thedir}/movie1/ and edit makemovie.sh and __init__.local.py if necessary and run makemovie.sh 1 1 1




# Notes:
# thickdisk7:
# 0267 and 2292 are bad for thickdisk7, but newer one exists that overwrites it.


