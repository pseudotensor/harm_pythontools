
# 1) Ensure sizes all correctly large
ls -alRS | grep fieldline | grep -v "\->" | sort -nrk 5

# 2) If not all large, then create list of dirs in order of simulation so overwrites use newer file
ls > thickdiskdirs.txt
emacs thickdiskdirs.txt  &

# 3) make links
for mydir in `cat ../thickdiskdirs.txt` ; do echo $mydir ; for fil in `ls ../$mydir/dumps/fieldline*.bin` ; do echo $fil ; ln -sf $fil . ;  done ; done

# 0267 and 2292 are bad for thickdisk7, but newer one exists that overwrites it.