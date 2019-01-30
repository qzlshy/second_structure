# second_structure
A context model to prediction protein secondary structure.<br>
The *.npy files in train_context_cullpdb are date set we used. They are uploaded with lfs, so you should install git-lfs to download *.npy files.<br>
The *.npy files can be loaded with python as:<br>
```
import numpy as np
data=np.load('*.npy').item()
name=data['name']
seq=data['seq']
pssm=data['pssm']
dssp=data['dssp']
```
Name is the protein structures name, seq is the sequence of structures, pssm is the psi-blasts profiles of sequences, dssp is the second structure of sequcences.<br>
