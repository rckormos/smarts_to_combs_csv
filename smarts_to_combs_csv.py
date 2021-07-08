import os
import sys
import gzip
import argparse
import multiprocessing

from io import StringIO
from itertools import chain as iterchain
from itertools import permutations as permute

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from prody import *
from scipy.spatial.transform import Rotation

RDLogger.DisableLog('rdApp.*')

def worker(tup):
    ligpdb, res_cutoff, robs_cutoff, match_mol, smarts_str, ref_coords = tup
    try:
        ligpdb.read_pdb()
        if not ligpdb.removed and (ligpdb.resolution > res_cutoff
                or ligpdb.r_obs > robs_cutoff):
            return None
        ref_mol = \
            Chem.rdmolops.RemoveHs(match_mol)
        ligpdb.set_dataframe(smarts_str, ref_coords, ref_mol)
        return ligpdb.dataframe
    except:
        return None

class PDBSmarts:
    """
    SMARTS pattern(s) with associated PDB search functionality.

    ...

    Attributes
    ----------
    workdir : os.path.realpath
        Directory in which to create final CSV file.
    smarts : list
        List of SMARTS patterns for which matches in the PDB were found.
    match_mols : dict
        Dict pairing each SMARTS pattern with a list of RDKit molecule 
        objects representing ligands that match that pattern.
    match_pdbs : dict
        Dict pairing each SMARTS pattern with a list of PDB accession codes 
        of structures containing a ligand that matches each SMARTS pattern.
    ref_coords : dict
        Dict pairing each SMARTS pattern with a list of arrays of reference 
        coordinates of the atoms that match the pattern in the ideal geometry. 

    Methods
    -------
    count_pdbs(smarts_str)
        Count the number of PDBs that match a given SMARTS pattern.
    read_matches(smarts_str, res_cutoff=2., robs_cutoff=0.3,
                 threads=1, save_memory=True)
        Read gzipped PDB files for PDB structures that match a given 
        SMARTS pattern, have 30% homology for all chains, and optionally 
        satisfy further cutoff criteria.
    write_combs_csv(smarts_str, res_quotient=2., robs_quotient=0.3)
        Write a CSV file for input to COMBS with all PDB matches to a given 
        SMARTS pattern.
    """
    def __init__(self, smarts, workdir=None, pdb_sdf_path=None, 
                 lig_list_path=None, pdb_clust_path=None, 
                 mirror_path=None, molprobity_path=None):
        self.smarts = smarts
        self.workdir = os.getcwd()
        # http://ligand-expo.rcsb.org/dictionaries/cc-to-pdb.tdd
        lig_list = '/home/kormos/thesis_project/pdb_ligands/cc-to-pdb.tdd'
        # http://ligand-expo.rcsb.org/dictionaries/Components-pub.sdf
        pdb_sdf = '/home/kormos/thesis_project/pdb_ligands/Components-pub.sdf'
        # https://cdn.rcsb.org/resources/sequence/clusters/bc-50.out
        pdb_clust = '/home/kormos/thesis_project/pdb_seq_clusters/bc-50.out'
        # rsync -rlpt -v -z --delete --port=33444 
        # rsync.rcsb.org::ftp_data/structures/divided/pdb/ <DEST FOLDER>
        mirror = '/home/kormos/pdb'
        if workdir:
            self.workdir = workdir
        if lig_list_path:
            lig_list = lig_list_path
        if pdb_sdf_path:
            pdb_sdf = pdb_sdf_path
        if pdb_clust_path:
            pdb_clust = pdb_clust_path
        if mirror_path:
            mirror = mirror_path
        self.molprobity_path = molprobity_path
        # read ligand list file
        with open(lig_list, 'r') as f:
            lines = f.read().split('\n')
        if lines[-1] == '':
            lines = lines[:-1]
        list_names = {line.split('\t')[0] for line in lines}
        # get the list of ligand molecules from the Ligand Expo SDF
        suppl = Chem.SDMolSupplier(pdb_sdf, removeHs=False)
        mols = []
        names = []
        pdbs = []
        for mol in suppl:
            try:
                Chem.SanitizeMol(mol)
                if mol.GetProp("_Name") in list_names:
                    names.append(mol.GetProp("_Name"))
                    mols.append(mol)
            except:
                pass
        # get list of PDBs containing each ligand in dict form
        pdbs = dict([(lig, plist.split()) for lig, plist in 
                     [line.split('\t') for line in lines] 
                     if lig in names])
        # get list of PDBs containing each SMARTS pattern in dict form
        self.match_mols = {}
        self.match_pdbs = {}
        self.ref_coords = {}
        for smarts_str in self.smarts:
            patt = Chem.MolFromSmarts(smarts_str)
            self.match_mols[smarts_str] = []
            self.match_pdbs[smarts_str] = []
            self.ref_coords[smarts_str] = None
            for mol, name in zip(mols, names):
                if mol.HasSubstructMatch(patt):
                    self.match_mols[smarts_str].append(mol)
                    self.match_pdbs[smarts_str].append([
                        LigandPDB(pdb, mirror, name, pdb_clust) 
                        for pdb in pdbs[name]])
                    if self.ref_coords[smarts_str] is None:
                        # get reference coordinates for substructure matching
                        idxs = np.array(mol.GetSubstructMatch(patt))
                        coords = mol.GetConformers()[0].GetPositions()[idxs]
                        self.ref_coords[smarts_str] = coords
        self._dfs = {}       

    def count_pdbs(self, smarts_str):
        if smarts_str in self._dfs.keys():
            print('n_pdbs =', len([ent for ent in self._dfs[smarts_str] 
                                   if ent is not None]))
        else:
            print('n_pdbs =', 
                  sum([len([p for p in plist if not p.removed]) 
                       for plist in self.match_pdbs[smarts_str]]))

    def read_matches(self, smarts_str, res_cutoff=2., robs_cutoff=0.3, 
                     molprobity_cutoff=2., threads=1, save_memory=True):
        if threads > 1:
            nmols = len(self.match_mols[smarts_str])
            match_mols = \
                iterchain.from_iterable([[self.match_mols[smarts_str][i]] * 
                                         len(self.match_pdbs[smarts_str][i]) 
                                         for i in range(nmols)])
            match_pdbs = \
                iterchain.from_iterable(self.match_pdbs[smarts_str])
            ref_coords = self.ref_coords[smarts_str]
            pool = multiprocessing.Pool(threads)
            self._dfs[smarts_str] = list(pool.imap(worker, 
                ((match_pdb, res_cutoff, robs_cutoff, match_mol, smarts_str, 
                ref_coords) for match_pdb, match_mol in 
                zip(match_pdbs, match_mols))))
            return
        self._dfs[smarts_str] = []
        for i, plist in enumerate(self.match_pdbs[smarts_str]):
            plist = self.match_pdbs[smarts_str][i]
            for j, ligpdb in enumerate(plist):
                if ligpdb.removed and save_memory:
                    self.match_pdbs[smarts_str][i][j] = None
                try:
                    ligpdb.read_pdb()
                    if not ligpdb.removed and (ligpdb.resolution > res_cutoff
                            or ligpdb.r_obs > robs_cutoff):
                        ligpdb.removed = True
                        if save_memory:
                            self.match_pdbs[smarts_str][i][j] = None
                        continue
                    ref_mol = \
                        Chem.rdmolops.RemoveHs(self.match_mols[smarts_str][i])
                    ligpdb.set_dataframe(smarts_str, 
                                         self.ref_coords[smarts_str], ref_mol)
                except:
                    ligpdb.removed = True
                    if save_memory:
                        self.match_pdbs[smarts_str][i][j] = None
                if not ligpdb.removed:
                    self._dfs[smarts_str].append(ligpdb.dataframe)
                    if save_memory:
                        self.match_pdbs[smarts_str][i][j] = None

    def write_combs_csv(self, smarts_str, res_quotient=2., robs_quotient=0.3):
        df = pd.concat([df for df in self._dfs[smarts_str] if df is not None])
        # rank homologous chains before outputting CSV
        clusters = set(df['cluster'])
        for cluster in clusters:
            cluster_df = df[df.cluster == cluster]
            structs = set(cluster_df['pdb_accession'])
            if len(structs) > 0:
               res = np.array(
                   [cluster_df[cluster_df.pdb_accession == 
                               struct]['res'].iloc[0] for struct in structs])
               r_obs = np.array(
                   [cluster_df[cluster_df.pdb_accession == 
                               struct]['r_obs'].iloc[0] for struct in structs])
               score = res / res_quotient + r_obs / robs_quotient
               ranks = len(score) - np.argsort(
                   np.argsort(score, kind='mergesort'), kind='mergesort')
               for struct, rank in zip(structs, ranks):
                   df.loc[(df.pdb_accession == struct) & 
                          (df.cluster == cluster), "cluster_rank"] = rank
        # output final CSV
        df = df.drop(['res', 'r_obs'], axis=1).astype({'resnum' : int,
                                                       'cluster' : int, 
                                                       'cluster_rank' : int})
        df.to_csv(self.workdir + '/{}.csv'.format(smarts_str))

                

class LigandPDB:
    """PDB structure containing a ligand and relevant functionality.

    Attributes
    ----------
    id : str
        Four-letter RCSB accession code for the structure.
    ligname : str
        Three-letter RCSB accession code for the bound ligand.
    removed : bool
        Bool denoting whether or not the PDB file has been removed.
    pdb_path : str
        Path to gzipped PDB file containing the structure.
    atoms : prody.AtomGroup
        ProDy AtomGroup of the biological assembly of the structure.
    resolution : float
        Resolution (in Angstroms) of the structure.
    r_obs : float
        Observed R value of the structure.
    dataframe : pandas.DataFrame
        COMBS-compatible dataframe containing molecular fragments from 
        the ligand(s) that match a SMARTS pattern.

    Methods
    -------
    read_pdb()
        Use ProDy to parse the corresponding gzipped PDB file and structure, 
        as well as resolution and r_obs.
    analyze()
        Identify residue number(s) and chain ID(s) for the ligand(s), as well 
        as chain IDs and homology clusters for contacting protein chains. 
    set_dataframe(smarts, ref_coords=None, ref_mol=None)
        Generate a COMBS-compatible dataframe containing molecular fragments 
        from the ligand(s) that match a SMARTS pattern as well as optional 
        reference coordinates against which to align by permuting atom order 
        and an optional reference molecule for determining bond order.
    """
    def __init__(self, pdb_id, mirror_dir, ligname, pdb_clust):
        self.id = pdb_id
        self.ligname = ligname
        self.removed = False
        self._mirror = mirror_dir
        self._pdb_clust = pdb_clust

    def read_pdb(self):
        self.pdb_path = self._mirror + '/' + self.id[1:3] + '/pdb' + \
                        self.id + '.ent.gz' 
        self._atoms = None
        self._header = None
        self.resolution = None
        self.r_obs = None
        asym, self._header = parsePDB(self.pdb_path, header=True)
        with gzip.open(self.pdb_path, 'rt') as f:
            lines = f.read().split('\n')
        if not os.path.exists(self.pdb_path):
            self.removed = True
            return
        self.resolution = self._header['resolution']
        self._atoms = buildBiomolecules(self._header, asym)
        if isinstance(self._atoms, list):
            for biol in self._atoms:
                if biol.select('resname ' + self.ligname):
                    self._atoms = biol
                    break
            if isinstance(self._atoms, list):
                self.removed = True
                return
        for line in lines:
            if not self.r_obs and 'REMARK   3   R VALUE' in line:
                self.r_obs = float([val for val in line.split() 
                                    if val != ''][-1])
        if None in [self.resolution, self.r_obs]:
            self.removed = True

    def set_dataframe(self, smarts, ref_coords=None, ref_mol=None):
        if self.removed:
            return
        self._set_chains_clusters()
        if not self._resnums:
            self.removed = True
            return
        self._set_names_coords(smarts, ref_coords, ref_mol)
        df = {'pdb_accession' : [], 'lig_chain' : [], 
              'prot_chain' : [], 'resnum' : [], 'resname' : [], 
              'name' : [], 'generic_name' : [], 
              'c_x' : [], 'c_y' : [], 'c_z' : [], 
              'cluster' : [], 'cluster_rank' : [], 
              'res' : [], 'r_obs' : []}
        for i in range(len(self._resnums)):
            for j in range(len(self._prot_chains[i])):
                for k in range(len(self._names[i])):
                    df['pdb_accession'].append(self.id)
                    df['lig_chain'].append(self._lig_chains[i])
                    df['prot_chain'].append(self._prot_chains[i][j])
                    df['resnum'].append(self._resnums[i])
                    df['resname'].append(self.ligname)
                    df['name'].append(self._names[i][k])
                    df['generic_name'].append('atom' + str(k))
                    df['c_x'].append(self._coords[i][k][0])
                    df['c_y'].append(self._coords[i][k][1])
                    df['c_z'].append(self._coords[i][k][2])
                    df['cluster'].append(self._clusters[i][j])
                    df['cluster_rank'].append(1) # ranking occurs later
                    df['res'].append(self.resolution)
                    df['r_obs'].append(self.r_obs)
        self.dataframe = pd.DataFrame(df)
        # free memory from self._atoms and self._header
        self._atoms, self._header = None, None

    def _set_chains_clusters(self):
        assert self.pdb_path # must be run after read_pdb()
        lig_sel = 'resname {}'.format(self.ligname)
        lig_atoms = self._atoms.select(lig_sel)
        if lig_atoms:
            self._resnums = list(set([a.getResnum() for a in 
                                      self._atoms.select(lig_sel)]))
        else:
            self._resnums = []
            return
        self._lig_chains = []
        self._lig_segi = []
        self._prot_chains = []
        self._clusters = []
        with open(self._pdb_clust, 'r') as f:
            lines = f.readlines()
        for num in self._resnums:
            lig_sel = 'resnum {}'.format(num)
            prot_sel = 'protein within 5 of ' + lig_sel
            if lig_sel is not None and prot_sel is not None:
                lig_atoms = self._atoms.select(lig_sel)
                prot_atoms = self._atoms.select(prot_sel)
            else: 
                self._resnums = [n for n in self._resnums if n != num]
            if prot_atoms:
                self._lig_chains.append([a.getChid() for a in lig_atoms][0])
                self._lig_segi.append([a.getSegindex() for a in lig_atoms][0])
                self._prot_chains.append(list(set([a.getChid() 
                                                   for a in prot_atoms])))
                self._clusters.append([])
                for chain in self._prot_chains[-1]:
                    full_id = self.id.upper() + '_' + chain
                    # find the index of the line which contains the chain ID
                    # if the chain is a peptide without a cluster, strike it
                    has_id = [(full_id + ' ' in line or 
                               full_id + '\n' in line) for line in lines]
                    if True in has_id:
                        self._clusters[-1].append(has_id.index(True))
                    else:
                        self._prot_chains[-1] = [c for c in 
                                                 self._prot_chains[-1] 
                                                 if c != chain]
            else:
                self._resnums = [n for n in self._resnums if n != num]
            
    def _set_names_coords(self, smarts, ref_coords=None, ref_mol=None):
        assert self._resnums # must be run after _set_chains_clusters()
        self._names = []
        self._coords = []
        for i, num in enumerate(self._resnums):
            selstr = 'resnum {} and resname {} and chain {} and segindex {}'
            sel = self._atoms.select(selstr.format(num, self.ligname, 
                                                   self._lig_chains[i], 
                                                   self._lig_segi[i]))
            # read ligand into RDKit
            pdbio = StringIO()
            writePDBStream(pdbio, sel)
            block = pdbio.getvalue()
            # not all ligands have hydrogen, so MolFromPDBBlock removes any H 
            # if it happens to be present, and then H is added back by RDKit
            try:
                mol = Chem.rdmolfiles.MolFromPDBBlock(block)
                if ref_mol is not None and mol.GetNumAtoms() == \
                        ref_mol.GetNumAtoms():
                    mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, mol) 
                mol = Chem.rdmolops.AddHs(mol, addCoords=True)
            except:
                self._resnums = [n for n in self._resnums if n != num]
                continue
            mol_names = []
            for line in Chem.rdmolfiles.MolToPDBBlock(mol).split('\n'):
                mol_names.append(line[12:16].strip())
            mol_elem = [a.GetSymbol() for a in mol.GetAtoms()]
            patt = Chem.MolFromSmarts(smarts)
            idxs_tup = mol.GetSubstructMatches(patt)
            if not idxs_tup:
                self._resnums = [n for n in self._resnums if n != num]
                continue
            for idxs in idxs_tup:
                pos = mol.GetConformers()[0].GetPositions()
                n_array = np.array(mol_names)[np.array(idxs)]
                e_array = np.array(mol_elem)[np.array(idxs)]
                # find best alignment of pos to ref_coords
                if ref_coords is not None:
                    mean_pos = np.mean(pos[np.array(idxs)], axis=0)
                    mean_ref = np.mean(ref_coords, axis=0)
                    perms = [np.array(p) for p in permute(range(len(idxs)))]
                    min_rmsd = np.inf
                    min_perm = perms[0]
                    for p in perms:
                        if not np.all(e_array[p] == e_array) or \
                                np.all(pos[np.array(idxs)] - 
                                ref_coords < 1e-6):
                            continue
                        rot, rmsd = Rotation.align_vectors(
                            pos[np.array(idxs)][p] - mean_pos, 
                            ref_coords - mean_ref)
                        if rmsd < min_rmsd:
                            min_rmsd = rmsd
                            min_perm = p
                    self._names.append(list(n_array[min_perm]))
                    self._coords.append(pos[np.array(idxs)][min_perm])
                else:
                    self._names.append(list(n_array))
                    self._coords.append(pos[np.array(idxs)])
                    


def parse_args():
    argp = argparse.ArgumentParser(description="Generate a COMBS-compatible" 
                                               "CSV given a SMARTS pattern.")
    argp.add_argument('smarts', nargs='+', help="SMARTS pattern of fragment "
                      "to be searched for in PDB ligands.")
    argp.add_argument('--dryrun', action='store_true', help="Do not read "
                      "any files, but count the number of matches.")
    argp.add_argument('-t', '--threads', type=int, default=1, help="Number "
                      "of threads on which to run the reading of PDB files.")
    args = argp.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    pdb_smarts = PDBSmarts(args.smarts)
    for smarts in pdb_smarts.smarts:
        pdb_smarts.count_pdbs(smarts)
        if not args.dryrun:
            pdb_smarts.read_matches(smarts, threads=args.threads)
            pdb_smarts.count_pdbs(smarts)
            pdb_smarts.write_combs_csv(smarts)
