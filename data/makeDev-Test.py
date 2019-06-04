#creates set of 100 original photos for hyperpparam tuning
outfile = open('originals100_dev.tsv','w')
i = 0
with open('originals_small_unprocessed.tsv', 'r') as f:
	for line in f:
		if(i % 30 == 0):
			line = line.strip()
			outfile.write(line+'\n')
		i+=1

#set for of 100 photoshopped images for hhyperparam tuning
outfile2 = open('photoshops100_dev.tsv','w')
i = 0
with open('photoshops_small_unprocessed.tsv', 'r') as f:
	for line in f:
		if(i % 30 == 0):
			line = line.strip()
			outfile2.write(line+'\n')
		i+=1
