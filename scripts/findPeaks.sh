#!/usr/bin/env bash

#-ctl_tag1 ${4} \
       
bds ${1}/chipseq.bds \
       -out_dir ${2} \
       -histone \
       -tag1 ${3} \
       -callpeak macs2 \
       -species ${4} \
       -nth 2


