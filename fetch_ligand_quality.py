import sys
import requests
import multiprocessing

import pandas as pd

def worker(tup):
    quality = {'pdb_accession' : [], 'ligand_name' : [], 
               'chain' : [], 'resnum' : [], 'quality' : []}
    pdb_id, lig_list = tup
    for lig_id in lig_list:
        url = 'http://rcsb.org/ligand-validation/{}/{}'.format(pdb_id.upper(), lig_id.upper()) 
        try:
            response = requests.get(url) 
            table_id = response.text.index('<table id="ligand-validation">') 
        except:  
            continue
        table_end_id = response.text[table_id:].index('</table>') + 8 
        html = response.text[table_id:][:table_end_id] 
        df = pd.read_html(html)[0] 
        query = pdb_id.upper() + '_' + lig_id.upper() 
        df = df[df['Identifier'].str.contains(query)] 
        pdb = [pdb_id.upper()] * len(df) 
        lig = [lig_id.upper()] * len(df) 
        chain = [val.split('_')[2] for val in df['Identifier']] 
        resnum = [val.split('_')[3] for val in df['Identifier']] 
        qual = list(df['Ranking for goodness of fit']) 
        quality['pdb_accession'] += pdb
        quality['ligand_name'] += lig
        quality['chain'] += chain
        quality['resnum'] += resnum
        quality['quality'] += qual
    return pd.DataFrame.from_dict(quality)

if __name__ == "__main__":
    infile = sys.argv[1] # cc-to-pdb.tdd
    outfile = sys.argv[2]

    with open(infile, 'r') as f:
        lines = [line.split('\t') for line in f.read().split('\n')[:-1]]
    indict = {line[0] : line[1].split() for line in lines}
    print('Inverting ligand-to-pdb mapping.')
    inv_dict = {}
    for key, value in indict.items():
        for entry in value:
            if not entry in inv_dict.keys():
                inv_dict[entry] = []
            inv_dict[entry].append(key)
    print('inverse dict length :', len(inv_dict))
    print('Making requests.')
    pool = multiprocessing.Pool(6)
    dfs = pool.imap(worker, tuple(inv_dict.items()))
    pd.concat([df for df in dfs if len(df) > 0]).to_csv(outfile)
