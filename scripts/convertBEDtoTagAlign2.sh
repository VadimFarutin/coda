#!/usr/bin/env bash
# Takes in a BED5 file from SE reads
# and outputs tagAlign

BEDPath=$1
tagAlignPath=$2

awkProg='
BEGIN {OFS = "\t"}
{
	printf "%s\t%s\t%s\tN\t1000\t+\n",$1,$2,$3
}
'

awk -F'\t' "${awkProg}" ${BEDPath}| \
	gzip -c > ${tagAlignPath}
