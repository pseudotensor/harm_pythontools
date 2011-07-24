
# On ki-jmck:

# 1) Ensure sizes all correctly large
ls -alRS thickdisk11* | grep fieldline | grep -v "\->" | sort -nrk 5

# 2) If not all large, then order list of dirs in order of simulation so overwrites use newer file and repeat

# 3) setup directory list

cd /data2/jmckinne/
ls > thickdiskdirs.txt
emacs thickdiskdirs.txt  &

# 4) Edit dir list and choose one's want

# 5) create new full-sim dir and change to dumps dir

mkdir thickdisk11/
cd thickdisk11/
mkdir dumps
cd dumps

# 6) make links
for mydir in `cat ../thickdiskdirs.txt` ; do echo $mydir ; for fil in `ls ../$mydir/dumps/fieldline*.bin` ; do echo $fil ; ln -sf $fil . ;  done ; done

# 7) Also make links to gdump.bin and dump0000.bin




# thickdisk7:
# 0267 and 2292 are bad for thickdisk7, but newer one exists that overwrites it.


