#!/usr/bin/env bash

#-ctl_tag1 ${4} \
       
bds ${1}/chipseq.bds \
       -out_dir ${2} \
       -histone \
       -tag1 ${3} \
       -callpeak macs2 \
       -final_stage peak \
       -species ${4} \
       -gensz "mm" \
       -chrsz "./data/mm9.male.chrom.sizes" \
       -nth 2


