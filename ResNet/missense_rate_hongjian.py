#!/usr/bin/env python
import os
import pysam
from collections import defaultdict
import gzip
MPC = '/home/local/ARCS/hq2130/missense_rate/fordist_constraint_official_mpc_values.txt.gz' 
with gzip.open(MPC) as f:
    head = f.readline()
    mpc1, mpc2 = set(), set()
    for line in f:
        line = line.strip().split('\t')
        chrom, pos, ref, alt = line[:4]
        score = line[-1]
        if score != 'NA':
            var_id = '_'.join([chrom, pos, ref, alt])
            #print var_id, score
            if float(score) >=1 :
                mpc1.add(var_id)
            if float(score) >=2 :
                mpc2.add(var_id)

revel = '/home/local/ARCS/hq2130/missense_rate/revel_all_chr.txt.gz'
with gzip.open(revel) as f:
    head = f.readline()
    revel4, revel5, revel6, revel7, revel8, revel9 = set(), set() , set(), set() ,set(), set()
    for line in f:
        chrom, pos, _, ref, alt,  score = line.strip().split('\t')
        if score != 'NA':
            var_id = '_'.join([chrom, pos, ref, alt])
            if float(score) >=0.4 :
                revel4.add(var_id)
            if float(score) >=0.5 :
                revel5.add(var_id)
            if float(score) >=0.6 :
                revel6.add(var_id)
            if float(score) >=0.7 :
                revel7.add(var_id)
            if float(score) >=0.8 :
                revel8.add(var_id)
            if float(score) >=0.9 :
                revel9.add(var_id)

mvp = '/home/local/ARCS/hq2130/missense_rate/All_rare_missense0724.txt'
with open(mvp) as f:
    head = f.readline()
    mvp005, mvp01, mvp02, mvp03, mvp04, mvp05, mvp06, mvp07, mvp08, mvp09 = set(), set() , set(), set() ,set(), set(), set(), set(), set(), set()
    for line in f:
        chrom, pos, ref, alt, _, score = line.strip().split('\t')
        if score != 'NA':
            var_id = '_'.join([chrom, pos, ref, alt])
            if float(score) >=0.05:
                mvp005.add(var_id)
            if float(score) >=0.1 :
                mvp01.add(var_id)
            if float(score) >=0.2 :
                mvp02.add(var_id)
            if float(score) >=0.3 :
                mvp03.add(var_id)
            if float(score) >=0.4 :
                mvp04.add(var_id)
            if float(score) >=0.5 :
                mvp05.add(var_id)
            if float(score) >=0.6 :
                mvp06.add(var_id)
            if float(score) >=0.7 :
                mvp07.add(var_id)
            if float(score) >=0.8 :
                mvp08.add(var_id)
            if float(score) >=0.9 :
                mvp09.add(var_id)

with open('enst.txt') as f:
    enst = set(line.strip().split()[0] for line in f.readlines())

fasta_file = '//home/local/ARCS/hq2130/Exome_Seq/resources/hg19.fasta'
dir1 = '/home/local/ARCS/hq2130/dbNSFPv3/'

def get_dna(chrom, pos):
    fastafile = pysam.Fastafile(fasta_file)
    seq = fastafile.fetch(chrom, pos-2, pos + 1).upper()    
    return seq

felix = defaultdict(lambda: 0)
with open('3mer_table_from_felix.txt') as f:
    head = f.readline()
    for line in f:
        lst = line.strip().split()
        felix[lst[0]+'->'+lst[1]] = float(lst[-1])

col_dict={'cadd15':('CADD_phred', 15), 'cadd20':('CADD_phred', 20),
          'cadd25':('CADD_phred', 25), 'cadd30':('CADD_phred', 30),
          'eigen_pred10':('Eigen-phred', 10), 'eigen_pred15':('Eigen-phred', 15),
          'eigen_pc_pred10':('Eigen-PC-phred', 10), 
          'MetaSVM>0':('MetaSVM_rankscore', 0.82271),'MetaLR>0':('MetaLR_rankscore', 0.81122), 
          'M_CAP>0.025':('M-CAP_rankscore', 0.4815), 'M_CAP>0.05':('M-CAP_rankscore', 0.642), 
          'PP2-HVAR':('Polyphen2_HVAR_rankscore', 0.6280),'FATHMM':('FATHMM_converted_rankscore', 0.8235),
          'all_missense':('CADD_phred', 0.0)}

mu_rate = defaultdict(lambda:  defaultdict(lambda: 0) ) 
transcipt2gene = {}
for fname in os.listdir(dir1):
    if fname.startswith('dbNSFP3.3a_variant') and 'chrM' not in fname:
        print fname
        
        with open(dir1 + fname) as f:
            head = f.readline().strip().split('\t')
            for line in f:
                info = dict(zip(head, line.strip().split('\t')))
		if info['ExAC_AF']!='.':
			if float(info['ExAC_AF'])>10^-5:
				continue

		if info['aaref']  not in {'X', '.'} and info['aaalt'] not in {'.', 'X'}:
			    chrom, pos, ref, alt = info['hg19_chr'], info['hg19_pos(1-based)'], info['ref'], info['alt']
			    if pos != '.':
				tri_ref = get_dna(chrom, int(pos))
				tri_alt = tri_ref[0] + alt + tri_ref[-1]
				rate = felix[tri_ref + '->' + tri_alt]
				transcripts = info['Ensembl_transcriptid'].split(';')
				for transcript in transcripts:
				    if transcript in enst:
					transcipt2gene[transcript] = info['genename']
					for col_name, (col, threshold) in col_dict.items():    
					    score = info[col]
					    if score != '.':
						score = float(score)
						if score >= threshold:
						    mu_rate[transcript][col_name] += rate

					var_id = '_'.join([chrom, pos, ref, alt])
					if var_id in mpc1:
					    mu_rate[transcript]['mpc_1'] += rate
					if var_id in mpc2:
					    mu_rate[transcript]['mpc_2'] += rate

					if var_id in revel4:
					    mu_rate[transcript]['revel4'] += rate
					if var_id in revel5:
					    mu_rate[transcript]['revel5'] += rate
					if var_id in revel6:
					    mu_rate[transcript]['revel6'] += rate
					if var_id in revel7:
					    mu_rate[transcript]['revel7'] += rate
					if var_id in revel8:
					    mu_rate[transcript]['revel8'] += rate
					if var_id in revel9:
					    mu_rate[transcript]['revel9'] += rate

					if var_id in mvp005:
					    mu_rate[transcript]['mvp005'] += rate
					if var_id in mvp01:
					    mu_rate[transcript]['mvp01'] += rate
					if var_id in mvp02:
					    mu_rate[transcript]['mvp02'] += rate
					if var_id in mvp03:
					    mu_rate[transcript]['mvp03'] += rate
					if var_id in mvp04:
					    mu_rate[transcript]['mvp04'] += rate
					if var_id in mvp05:
					    mu_rate[transcript]['mvp05'] += rate
					if var_id in mvp06:
					    mu_rate[transcript]['mvp06'] += rate
					if var_id in mvp07:
					    mu_rate[transcript]['mvp07'] += rate
					if var_id in mvp08:
					    mu_rate[transcript]['mvp08'] += rate
					if var_id in mvp09:
					    mu_rate[transcript]['mvp09'] += rate
				    
 
	
            
with open('mis_rate_hongjian0725_excludeAF<10-5.txt', 'w') as fw:
    rates = col_dict.keys() + ['mpc_1', 'mpc_2', 'revel4', 'revel5', 'revel6','revel7','revel8','revel9', 
                               'mvp005', 'mvp01', 'mvp02', 'mvp03', 'mvp04', 'mvp05', 'mvp06', 'mvp07', 'mvp08',  'mvp09']
    print rates
    head = ['gene', 'transcript'] + rates
    fw.write('\t'.join(head) + '\n')
    for enst, rate_dict in mu_rate.items():
        info = map(str, [transcipt2gene[enst], enst] + [rate_dict[rate] for rate in rates])
        fw.write('\t'.join(info) + '\n')                
