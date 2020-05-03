#!/usr/bin/env bash
# Runs the BDS CHIP-seq pipeline on a given tagAlign file.

pipelineDir=$1
tagAlignPath=$2
outputDir=$3
species=$4

#    -final_stage xcor \
#    -subsample_xcor "10M" \

bds ${pipelineDir}/chipseq.bds \
    -out_dir ${outputDir} \
    -histone \
    -input tag \
    -final_stage tag \
    -tag1 ${tagAlignPath} \
    -tag2bw \
    -species ${species} \
    -nth 2
