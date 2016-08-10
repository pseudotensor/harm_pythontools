#!/bin/bash

list=`qstat -u jmckinn2|grep -v jonharmrad|awk '{print $1}' | tail -n +4`

for fil in $list ; do echo $fil ; qdel $fil ; done
