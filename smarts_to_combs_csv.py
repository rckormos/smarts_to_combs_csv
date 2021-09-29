import os
import sys
import gzip
import shutil
import argparse
import multiprocessing

from io import StringIO
from itertools import chain as iterchain
from itertools import permutations as permute

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import RDLogger
from prody import *
from scipy.spatial.transform import Rotation

RDLogger.DisableLog('rdApp.*')

def worker(tup):
    ligpdb, res_cutoff, robs_cutoff, match_mol, \
        smarts_str, discard, metals, ref_coords, perms = tup
    ligpdb.read_pdb()
    if not ligpdb.removed and (ligpdb.resolution > res_cutoff
            or ligpdb.r_obs > robs_cutoff):
        return None
    ref_mol = \
        Chem.rdmolops.RemoveHs(match_mol)
    ligpdb.set_dataframe(smarts_str, discard, metals, 
                         ref_coords, ref_mol, perms)
    return ligpdb.dataframe

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
    def __init__(self, smarts, discard=False, metals=None, workdir=None, 
                 lig_weight_limit=1000, tanimoto_threshold=0.8, 
                 pdb_sdf_path=None, lig_list_path=None, pdb_clust_path=None, 
                 mirror_path=None, molprobity_path=None):
        self.smarts = smarts
        self.discard = discard
        self.metals = metals
        self.workdir = os.getcwd()
        # http://ligand-expo.rcsb.org/dictionaries/cc-to-pdb.tdd
        lig_list = '/Users/kormos/smarts_to_combs_csv/cc-to-pdb.tdd'
        # http://ligand-expo.rcsb.org/dictionaries/Components-pub.sdf
        pdb_sdf = '/Users/kormos/smarts_to_combs_csv/Components-pub.sdf'
        # https://cdn.rcsb.org/resources/sequence/clusters/bc-50.out
        pdb_clust = '/Users/kormos/smarts_to_combs_csv/bc-50.out'
        # rsync -rlpt -v -z --delete --port=33444 
        # rsync.rcsb.org::ftp_data/structures/divided/pdb/ <DEST FOLDER>
        mirror = '/Users/kormos/pdb'
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
            if mol is not None:
                Chem.SanitizeMol(mol)
                if mol.GetProp("_Name") in list_names and \
                        ExactMolWt(mol) < lig_weight_limit:
                    names.append(mol.GetProp("_Name"))
                    mols.append(mol)
        # get list of PDBs containing each ligand in dict form
        pdbs = dict([(lig, plist.split()) for lig, plist in 
                     [line.split('\t') for line in lines] 
                     if lig in names])
        # get list of PDBs containing each SMARTS pattern in dict form
        self.match_mols = {}
        self.match_pdbs = {}
        self.ref_coords = {}
        self._perms = {}
        self._tanimoto_clusters = {}
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
                        if self.discard:
                             keep_idxs = [i for i, atom in 
                                          enumerate(patt.GetAtoms()) 
                                          if atom.GetAtomicNum() != 0]
                             coords = coords[keep_idxs]
                        self.ref_coords[smarts_str] = coords
            # get tanimoto clusters of ligands
            self.set_tanimoto_clusters(smarts_str, tanimoto_threshold)
            # get symmetry-equivalent permutations of atoms in patt
            n_atoms = patt.GetNumAtoms()
            patt.UpdatePropertyCache()
            sym_classes = np.array(Chem.CanonicalRankAtoms(patt, 
                                                           breakTies=False))
            dt = np.dtype([('', np.int64)] * n_atoms)
            perms = np.fromiter(permute(np.arange(n_atoms)), 
		                dt).view(np.int64).reshape(-1, n_atoms)
            sym_perms = sym_classes[perms]
            self._perms[smarts_str] = \
                perms[np.all(sym_perms == sym_perms[0], axis=1)]
        self._dfs = {}
        self._final_df = {}

    def count_pdbs(self, smarts_str):
        if smarts_str not in self._final_df.keys():
            print('n_ligands =', len(self.match_mols[smarts_str]))
        else:
            long_df = self._final_df[smarts_str]
            short_df = long_df[long_df['lig_cluster_rank'] == 1]
            print('n_ligands =', len(set(long_df['resname'])))
            print('n_dissimilar_ligands =', len(set(short_df['resname'])))
        if smarts_str in self._dfs.keys():
            print('n_pdbs =', len([ent for ent in self._dfs[smarts_str] 
                                   if ent is not None]))
        else:
            print('n_pdbs =', 
                  sum([len([p for p in plist if not p.removed]) 
                       for plist in self.match_pdbs[smarts_str]]))
            if len(self.match_mols[smarts_str]) > 1:
                print('Example Ligand :', 
                      self.match_mols[smarts_str][0].GetProp("_Name"))

    def read_matches(self, smarts_str, discard=False, metals=None,  
                     res_cutoff=2., robs_cutoff=0.3, molprobity_cutoff=2., 
                     threads=1, save_memory=True):
        nmols = len(self.match_mols[smarts_str])
        match_mols = \
            iterchain.from_iterable([[self.match_mols[smarts_str][i]] * 
                                     len(self.match_pdbs[smarts_str][i]) 
                                     for i in range(nmols)])
        match_pdbs = \
            iterchain.from_iterable(self.match_pdbs[smarts_str])
        ref_coords = self.ref_coords[smarts_str]
        if threads > 1:
            pool = multiprocessing.Pool(threads)
            self._dfs[smarts_str] = list(pool.imap(worker, 
                ((match_pdb, res_cutoff, robs_cutoff, match_mol, smarts_str, 
                discard, metals, ref_coords, self._perms[smarts_str]) 
                for match_pdb, match_mol in zip(match_pdbs, match_mols))))
        else:
            self._dfs[smarts_str] = []
            for match_pdb, match_mol in zip(match_pdbs, match_mols):
                self._dfs[smarts_str].append(worker((match_pdb, res_cutoff, 
                                                     robs_cutoff, match_mol, 
                                                     smarts_str, discard, 
                                                     metals, ref_coords, 
                                                     self._perms[smarts_str])))

    def write_combs_csv(self, smarts_str, res_quotient=2., robs_quotient=0.3):
        all_dfs = [df for df in self._dfs[smarts_str] if df is not None]
        if len(all_dfs):
            df = pd.concat(all_dfs)
        else:
            return
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
        # rank tanimoto-similar ligands before outputting CSV
        final_ligs = set(df['resname'])
        self._tanimoto_clusters[smarts_str] = \
            [[lig for lig in cluster if lig in final_ligs] for cluster 
             in self._tanimoto_clusters[smarts_str] 
             if len(final_ligs.intersection(cluster)) > 0]
        lig_clusters = {}
        lig_cluster_ranks = {}
        for lig in final_ligs:
            c_id = [i for i, cluster in 
                    enumerate(self._tanimoto_clusters[smarts_str]) 
                    if lig in cluster][0]
            lig_clusters[lig] = c_id + 1
            lig_cluster_ranks[lig] = \
                self._tanimoto_clusters[smarts_str][c_id].index(lig) + 1
        df['lig_cluster'] = [lig_clusters[lig] for lig in df['resname']]
        df['lig_cluster_rank'] = [lig_cluster_ranks[lig] for lig in 
                                  df['resname']]
        # output final CSV
        df = df.drop(['res', 'r_obs'], axis=1).astype({'lig_segi' : int,
                                                       'prot_segi' : int,
                                                       'resnum' : int,
                                                       'cluster' : int, 
                                                       'cluster_rank' : int})
        csv_name = smarts_str
        if self.metals:
            csv_name += '_' + '_'.join(self.metals)
        elif self.discard:
            csv_name += '_discard_wildcards'
        csv_name.replace('/', '')
        csv_name.replace('\\', '')
        df.to_csv(self.workdir + '/{}.csv'.format(csv_name))
        self._final_df[smarts_str] = df

    def set_tanimoto_clusters(self, smarts_str, tanimoto_threshold):
        names = [mol.GetProp("_Name") for mol in self.match_mols[smarts_str]]
        if tanimoto_threshold == 1.:
            self._tanimoto_clusters[smarts_str] = [[name] for name in names]
            return
        self._tanimoto_clusters[smarts_str] = [] 
        nmols = len(self.match_mols[smarts_str])
        print('Getting fingerprints.')
        fingerprints = [Chem.RDKFingerprint(mol) for mol in 
                        self.match_mols[smarts_str]]
        sim_array = np.ones((nmols, nmols))
        print('Computing pairwise Tanimoto scores.')
        for i, j in zip(*np.triu_indices(nmols, 1)):
            sim = FingerprintSimilarity(fingerprints[i], fingerprints[j])
            sim_array[i, j], sim_array[j, i] = sim, sim
        print('Clustering by Tanimoto score.')
        # greedily cluster ligands by tanimoto score
        gt_thresh = (sim_array > tanimoto_threshold).astype(int)
        n_neighbors = np.sum(gt_thresh, axis=0) - 1
        clustered = np.zeros(nmols).astype(bool)
        for i in np.argsort(n_neighbors)[::-1]:
            if not clustered[i]:
                cluster = np.argwhere(np.logical_and(gt_thresh[i], 
                                                     ~clustered)).flatten()
                clustered[cluster] = True
                cluster_names = [names[i] for i in cluster]
                self._tanimoto_clusters[smarts_str].append(cluster_names)

                

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
    set_dataframe(smarts, discard=False, metals=None, 
                  ref_coords=None, ref_mol=None)
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
        if not os.path.exists(self.pdb_path):
            self.removed = True
            return
        asym, self._header = parsePDB(self.pdb_path, header=True)
        with gzip.open(self.pdb_path, 'rt') as f:
            lines = f.read().split('\n')
        if 'resolution' in self._header.keys():
            self.resolution = self._header['resolution']
        try:
            self._atoms = buildBiomolecules(self._header, asym)
        except:
            self.removed = True
            return
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
                r_obs_str = [val for val in line.split() if val != ''][-1]
                if set('.123456789').intersection(r_obs_str):
                    self.r_obs = float(r_obs_str)
        if None in [self.resolution, self.r_obs]:
            self.removed = True

    def set_dataframe(self, smarts, discard=False, metals=None, 
                      ref_coords=None, ref_mol=None, perms=None):
        if self.removed:
            self.dataframe = None
            return
        self._set_chains_clusters(metals)
        if not self._resnums:
            self.dataframe = None
            self.removed = True
            return
        self._set_names_coords(smarts, discard, ref_coords, ref_mol, perms)
        df = {'pdb_accession' : [], 'lig_chain' : [], 'lig_segi' : [],  
              'prot_chain' : [], 'prot_segi' : [], 'resnum' : [], 
              'resname' : [], 'name' : [], 'generic_name' : [], 
              'c_x' : [], 'c_y' : [], 'c_z' : [], 
              'cluster' : [], 'cluster_rank' : [], 
              'res' : [], 'r_obs' : []}
        for i in range(len(self._resnums)):
            for j in range(len(self._prot_chains[i])):
                for k in range(len(self._names[i])):
                    df['pdb_accession'].append(self.id)
                    df['lig_chain'].append(self._lig_chains[i])
                    df['lig_segi'].append(self._lig_segi[i])
                    df['prot_chain'].append(self._prot_chains[i][j])
                    df['prot_segi'].append(self._prot_segi[i][j])
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
                if metals is not None:
                    df['pdb_accession'].append(self.id)
                    df['lig_chain'].append(self._metal_chains[i])
                    df['lig_segi'].append(self._metal_segi[i])
                    df['prot_chain'].append(self._prot_chains[i][j])
                    df['prot_segi'].append(self._prot_segi[i][j])
                    df['resnum'].append(self._metal_resnums[i])
                    df['resname'].append(self._metal_names[i])
                    df['name'].append(self._metal_names[i])
                    df['generic_name'].append('atom' + 
                                              str(len(self._names[i])))
                    df['c_x'].append(self._metal_coords[i][0])
                    df['c_y'].append(self._metal_coords[i][1])
                    df['c_z'].append(self._metal_coords[i][2])
                    df['cluster'].append(self._clusters[i][j])
                    df['cluster_rank'].append(1) # ranking occurs later
                    df['res'].append(self.resolution)
                    df['r_obs'].append(self.r_obs)
        self.dataframe = pd.DataFrame(df)
        # free memory from self._atoms and self._header
        self._atoms, self._header = None, None

    def _set_chains_clusters(self, metals=None):
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
        self._metal_resnums = []
        self._metal_names = []
        self._metal_chains = []
        self._metal_segi = []
        self._metal_coords = []
        self._prot_chains = []
        self._prot_segi = []
        self._clusters = []
        with open(self._pdb_clust, 'r') as f:
            lines = f.readlines()
        for num in self._resnums:
            lig_sel = 'resnum {}'.format(num)
            lig_atoms = self._atoms.select(lig_sel)
            prot_sel = 'protein within 5 of ' + lig_sel
            if metals:
                metals_or = ' or resname '.join(metals)
                metal_sel = 'resname {} within 5 of '.format(metals_or) + lig_sel
                sel_atoms = self._atoms.select(metal_sel)
                if sel_atoms:
                    # find closest metal to the ligand
                    dists = measure.measure.buildDistMatrix(lig_atoms, sel_atoms, 
                                                            format='arr')
                    min_idx = np.argmin(np.min(dists, axis=0))
                    metal_atom = list(sel_atoms)[min_idx]
                    metal_num = metal_atom.getResnum()
                    metal_name = metal_atom.getResname()
                    prot_sel += ' and protein within 5 of resnum {}'.format(
                        metal_num)
                    prot_atoms = self._atoms.select(prot_sel)
                    if prot_atoms:
                        self._metal_resnums.append(metal_num)
                        self._metal_names.append(metal_name)
                        self._metal_chains.append(metal_atom.getChid())
                        self._metal_segi.append(metal_atom.getSegindex())
                        self._metal_coords.append(metal_atom.getCoords())
                else:
                    self._resnums = [n for n in self._resnums if n != num]
                    continue
            else:
                prot_atoms = self._atoms.select(prot_sel)
            if prot_atoms:
                self._lig_chains.append([a.getChid() for a in lig_atoms][0])
                self._lig_segi.append([a.getSegindex() for a in lig_atoms][0])
                all_chains = [a.getChid() for a in prot_atoms]
                all_segi = [a.getSegindex() for a in prot_atoms]
                self._prot_chains.append(list(set(all_chains)))
                self._prot_segi.append([all_segi[all_chains.index(chid)] 
                                        for chid in set(all_chains)])
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
                        tups = [(c, s) for c, s in 
                                zip(self._prot_chains[-1], 
                                    self._prot_segi[-1]) if c != chain]
                        if len(tups):
                            self._prot_chains[-1], self._prot_segi[-1] = \
                            zip(*tups)
                        else:
                            self._prot_chains[-1], self._prot_segi[-1] = [], []
            else:
                self._resnums = [n for n in self._resnums if n != num]
        self._metal_coords = np.array(self._metal_coords)
            
    def _set_names_coords(self, smarts, discard=False, ref_coords=None, 
                          ref_mol=None, perms=None):
        assert self._resnums # must be run after _set_chains_clusters()
        self._names = []
        self._coords = []
        for i, num in enumerate(self._resnums):
            selstr = 'resnum {} and resname {} and chain {} and segindex {}'
            sel = self._atoms.select(selstr.format(num, self.ligname, 
                                                   self._lig_chains[i], 
                                                   self._lig_segi[i]))
            if not sel:
                self._resnums = [n for n in self._resnums if n != num]
                continue
            # read ligand into RDKit
            pdbio = StringIO()
            writePDBStream(pdbio, sel)
            block = pdbio.getvalue()
            # not all ligands have hydrogen, so MolFromPDBBlock removes any H 
            # if it happens to be present, and then H is added back by RDKit
            mol = Chem.rdmolfiles.MolFromPDBBlock(block)
            try:
                if ref_mol is not None and mol.GetNumAtoms() == \
                        ref_mol.GetNumAtoms():
                    mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, mol)
            except:
                self._resnums = [n for n in self._resnums if n != num]
                continue
            mol = Chem.rdmolops.AddHs(mol, addCoords=True)
            mol_names = []
            for line in Chem.rdmolfiles.MolToPDBBlock(mol).split('\n'):
                mol_names.append(line[12:16].strip())
            patt = Chem.MolFromSmarts(smarts)
            idxs_tup = mol.GetSubstructMatches(patt)
            if not idxs_tup:
                self._resnums = [n for n in self._resnums if n != num]
                continue
            for idxs in idxs_tup:
                if discard:
                    keep_idxs = [i for i, atom in enumerate(patt.GetAtoms())
                                 if atom.GetAtomicNum() != 0]
                    idxs = np.array(idxs)[keep_idxs]
                else:
                    idxs = np.array(idxs)
                pos = np.round(mol.GetConformers()[0].GetPositions(), 3)
                n_array = np.array(mol_names)[idxs]
                # find best alignment of pos to ref_coords
                if ref_coords is not None and perms is not None:
                    mean_pos = np.mean(pos[idxs], axis=0)
                    mean_ref = np.mean(ref_coords, axis=0)
                    min_rmsd = np.inf
                    min_perm = perms[0]
                    for p in perms:
                        rot, rmsd = Rotation.align_vectors(
                            pos[idxs][p] - mean_pos, ref_coords - mean_ref)
                        if rmsd < min_rmsd:
                            min_rmsd = rmsd
                            min_perm = p
                    self._names.append(list(n_array[min_perm]))
                    self._coords.append(pos[idxs][min_perm])
                else:
                    self._names.append(list(n_array))
                    self._coords.append(pos[idxs]) 


def parse_args():
    argp = argparse.ArgumentParser(description="Generate a COMBS-compatible" 
                                               "CSV given a SMARTS pattern.")
    argp.add_argument('smarts', nargs='+', help="SMARTS pattern of fragment "
                      "to be searched for in PDB ligands.")
    argp.add_argument('--dryrun', action='store_true', help="Do not read "
                      "any files, but count the number of matches.")
    argp.add_argument('-t', '--threads', type=int, default=1, help="Number "
                      "of threads on which to run the reading of PDB files.")
    argp.add_argument('-d', '--discard', action='store_true', 
                      help="Discard wildcard atoms in the SMARTS pattern.")
    argp.add_argument('-m', '--metals', nargs='+', help="PDB two-letter "
                      "accession code(s) for possible metals to be included "
                      "in the search as a single extra atom, if desired.")
    argp.add_argument('-w', '--weight-limit', type=float, default=1000., 
                      help="Molecular weight limit on ligands to consider. "
                      "(Default: 1000 Daltons)")
    argp.add_argument('--tanimoto', type=float, default=1., 
                      help="Cluster threshold by which to cluster ligands.")
    args = argp.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    pdb_smarts = PDBSmarts(args.smarts, args.discard, args.metals, None,  
                           args.weight_limit, args.tanimoto)
    for smarts in pdb_smarts.smarts:
        pdb_smarts.count_pdbs(smarts)
        if not args.dryrun:
            pdb_smarts.read_matches(smarts, discard=args.discard, 
                                    metals=args.metals, threads=args.threads)
            pdb_smarts.write_combs_csv(smarts)
            pdb_smarts.count_pdbs(smarts)
