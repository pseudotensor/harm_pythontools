#!/bin/bash

# (nohup sh doallmakeallmovie.sh &)

export moviename="movienew2"

echo "STAGE1"
bash ./makeallmovie.sh ${moviename} 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 # setup links, copy files, and makeavg step that makes avg2d_*.npy by submitting job

sleep 40m # job takes 20

echo "STAGE2"
bash ./makeallmovie.sh ${moviename} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 # makeavgmerge: makes merged avg2d.npy

sleep 20m # 5 realistic

echo "STAGE3"
bash ./makeallmovie.sh ${moviename} 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 # make1d: makes qty2_*.npy files, submits job.

sleep 40m # job takes 20

echo "STAGE4"
bash ./makeallmovie.sh ${moviename} 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 # make1dmerge: merges to get qty2.npy file, local job (must come after make1d done)

sleep 20m # 5 realistic

echo "STAGE5"
bash ./makeallmovie.sh ${moviename} 0 0 0 0 0 100 0 1 1 1 1 1 1 0 0 0 0 0 0 # makeplot, submits 16core job and does all makeplot stuff in parallel

sleep 5m # for scripts to get done on head node

bash ./makeallmovie.sh ${moviename} 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 # makeframe: Make frames for movie, submits job.

sleep 5m # for scripts to get done on head node

bash ./makeallmovie.sh ${moviename} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 0 # makeavgplot, submits 16core job and does all makeavgplot stuff in parallel

sleep 40m # jobs take 20

echo "STAGE6"
bash ./makeallmovie.sh ${moviename} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 # collect all .  Creates ${moviename}/tbl*.tex that you can put into latex with \input{} for pdf.  This collects multiple runs together into single .tex file.

sleep 5m # for scripts to get done on head node

bash ./makeallmovie.sh ${moviename} 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # makemontage: local job (must come after makeplot done)

sleep 5m # for scripts to get done on head node

bash ./makeallmovie.sh ${moviename} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 # makemovie: create mp4 movie, local job.

sleep 5m # for scripts to get done on head node


sleep 20m # 5 realisic

echo "DONESTAGE"

