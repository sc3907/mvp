import sys



class MutationBackground(object):
    """
    MutationBackground class used to calcuate mutation background
    """
   

    def __init__(self, fname):
        """Return a MutationBackground object whose rate is a dict with each gene
        and its mutation background rate."""
        self.mut_type ={ 'all_missense',
                         'PP2-HVAR',
                         'FATHMM',
                         'eigen_pred10',
                         'eigen_pred15',
                         'MetaSVM>0',
                         'MetaLR>0',
                         'M_CAP>0.025',
                         'M_CAP>0.05',
                         'eigen_pc_pred10',
                         'cadd15',
                         'cadd20',
                         'cadd25',
                         'cadd30',
                         'mpc_1',
                         'mpc_2',
                         'revel4',
                         'revel5',
                         'revel6',
                         'revel7',
                         'revel8',
                         'revel9',
                         'mvp005',
                         'mvp01',
                         'mvp02',
                         'mvp03',
                         'mvp04',
                         'mvp05',
                         'mvp06',
                         'mvp07',
                         'mvp08',
                         'mvp09'}

        self.exclude_col = {'Gene', '#Reference', 'GeneVersion', 'Transcript.TranscriptVersion', 'ExonCount', 
                            'Chr', 'StartPos', 'EndPos', 'GeneName', 'gene', 'transcript'}

        self._init_rate(fname)

    def _convert(self, mutation_type):
        ''' used for convert some cols name difference from input file
        '''
        mutation_type_syn = {'revel4':'revel_0.4',
                             'revel5':'revel_0.5',
                             'revel6':'revel_0.6',
                             'revel7':'revel_0.7',
                             'revel8':'revel_0.8',
                             'revel9':'revel_0.9',
                             'mpc_1':'MPC>1', 
                             'mpc_2':'MPC>2',
                             'mvp005': 'cnn_0.05',   
                             'mvp01': 'cnn_0.1',   
                             'mvp02': 'cnn_0.2',   
                             'mvp03': 'cnn_0.3',   
                             'mvp04': 'cnn_0.4',   
                             'mvp05': 'cnn_0.5', 
                             'mvp06': 'cnn_0.6',  
                             'mvp07': 'cnn_0.7', 
                             'mvp08': 'cnn_0.8', 
                             'mvp09': 'cnn_0.9'}

        return mutation_type_syn.get(mutation_type, mutation_type)


    def _init_rate(self, fname):
        """inti mutation rate using Na's format, used the longest exon as final reslut."""
        self.rate = {}
        with open(fname) as f:
            head = f.readline().strip().split()
            for line in f:
                lst = line.strip().split()
                info = dict(zip(head, lst))
                gene = info['gene']
                rate = {}
                for mut_type in head:
                    if mut_type not in self.exclude_col:
                        mut_type_converted = self._convert(mut_type)
                        rate[mut_type_converted] = float(info[mut_type])
                        
                if gene not in self.rate:
                    self.rate[gene] = rate
                else:
                    print 'something wrong: gene occur twice', gene

    def expectation(self, geneset, mut_type, verbose=True):
        """Return the mutation background given a mutation rate dict."""
        exp = 0
        cur_mut_type = set(self.rate[self.rate.keys()[0]].keys())
        if mut_type in cur_mut_type:
            for gene in geneset:
                if gene in self.rate:
                    exp += float(self.rate[gene][mut_type]) * 2
        else:
            if verbose:
                print 'do not have rate for {}, use control instead'.format(mut_type)
        return exp
