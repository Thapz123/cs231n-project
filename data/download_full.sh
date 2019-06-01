#!/bin/bash

#helper function for downloading and checking files
download () { #url, destination, checksum
	if [ -f $2 ]; then
    echo "File $2 exists, skipping"
	return 
	fi
	
	#curl quiet, follow redirects, abort after 20 seconds
	curl -sS -L -m 20 -o $2 $1
	
	if [ ! $? -eq 0 ]; then
    echo "Error during download of $1" | tee -a error.log
	rm $2
	return
	fi
	
	sha=$(sha256sum $2 | cut -f 1 -d " ")
	
	if [ $sha != $3 ]; then
	echo "Error in checksum for $1, file $2" | tee -a error.log
	echo "for file $2, checksum $3 is correct"
	fi
}

#check for tsv files
[ ! -f originals_full_unprocessed.tsv ] && { echo "originals_full_unprocessed.tsv not found"; exit 127; }
[ ! -f photoshops_full_unprocessed.tsv ] && { echo "photoshops_full_unprocessed.tsv not found"; exit 127; }

#set up dirs
[ -d originals_full ] || mkdir originals_full
[ -d photoshops_full ] || mkdir photoshops_full

#Progress Reporting
originalsWC=$(wc -l < originals_full_unprocessed.tsv)
originalsWC=$(($originalsWC-1))
counter=0
echo "start downloading $originalsWC originals_full"
{
	read header #skip header
	while read -r id url end hash filesize score author link
	do
	  file="originals_full/$id.$end"
		download ${url} ${file} ${hash}
		counter=$((counter + 1))
		if ! ((counter % 100)); then
      echo "processed $counter/$originalsWC originals_full."
    fi
	done
}< originals_full_unprocessed.tsv

echo "finished downloading originals"
echo ""
echo "start downloading photoshops"

#Progress Reporting
photoshopWC=$(wc -l < photoshops_full_unprocessed.tsv)
photoshopWC=$((photoshopWC-1))
counter=0
{
	read header #skip header
	while read -r id original url end hash filesize score author link
	do
		file="photoshops_full/$id-$original.$end"
		counter=$((counter + 1))
		download ${url} ${file} ${hash}
		if ! ((counter % 100)); then
      echo "processed $counter/$photoshopWC photoshops_full."
    fi
	done
}< photoshops_full_unprocessed.tsv

echo "finished downloading photoshops_full"
