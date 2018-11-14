import numpy as np
from pyscf.fci import direct_uhf as fcisolver
import subprocess
import sys
'''
A simple 1-iteration DMET routine to test the dmrg solver.
'''
# system setup
norb = 200
nelec = norb
nimp = 2
nemb = 2*nimp
nelec_emb = nimp*2
U = float(sys.argv[1])
ibath = False
getrdm2 = True
maxm = 600
afm = float(sys.argv[2])

# construct lattice Hamiltonian
h1e  = np.zeros((2, norb, norb))
for i in range(norb):
    h1e[0][i,(i+1)%norb] = -1.
    h1e[0][i,(i-1)%norb] = -1.
    h1e[1][i,(i+1)%norb] = -1.
    h1e[1][i,(i-1)%norb] = -1.
    if(i%2==0):
        h1e[0][i,i] = afm
        h1e[1][i,i] = -afm
    else:
        h1e[1][i,i] = afm
        h1e[0][i,i] = -afm
        
# solve mean-field
ewa, eva = np.linalg.eigh(h1e[0])
ewb, evb = np.linalg.eigh(h1e[1])
dm1a = np.einsum('ij,kj-> ik', eva[:,:nelec/2],eva[:,:nelec/2].conj())
dm1b = np.einsum('ij,kj-> ik', evb[:,:nelec/2],evb[:,:nelec/2].conj())

# construct bath
Ra = np.zeros((norb,nimp*2))
Rb = np.zeros((norb,nimp*2))
Ra[:nimp,:nimp] = np.eye(nimp)
Rb[:nimp,:nimp] = np.eye(nimp)
_,_,ba = np.linalg.svd(dm1a[:nimp,nimp:],full_matrices=False)
_,_,bb = np.linalg.svd(dm1b[:nimp,nimp:],full_matrices=False)
Ra[nimp:,nimp:] = ba.conj().T
Rb[nimp:,nimp:] = bb.conj().T

# construct embedding Hamiltonian
h1emb = np.zeros((2, 2*nimp,2*nimp))
h1emb[0] = np.dot(Ra.conj().T, np.dot(h1e[0], Ra))
h1emb[1] = np.dot(Rb.conj().T, np.dot(h1e[1], Rb))

if ibath:
    Vemb = np.einsum('ip,iq,ir,is->pqrs',Ra.conj(),Rb.conj(),Ra,Rb)*U
else:
    Vemb = np.zeros((nemb,)*4)
    for i in range(nimp):
        Vemb[i,i,i,i] = U
#for i in range(nemb):
#    for j in range(nemb):
#        for k in range(nemb):
#            for l in range(nemb):
#                if abs(Vemb[i,j,k,l]) > 1e-11:
#                    print "%d %d %d %d: "%(i,j,k,l), Vemb[i,j,k,l]
# save txt
np.savetxt("./dump/hamfile.txt", h1emb.reshape(2*nemb,nemb))
np.savetxt("./dump/intfile.txt", Vemb.reshape(nemb**3,nemb))
np.savetxt("./dump/impsite.txt", np.arange(nimp)+1,fmt='%d')

# write input file
fin = open("input_solver", "w")
fin.write("input\n{\n")
fin.write("hamfile = ./dump/hamfile.txt\n")
fin.write("outdir = ./dump/\n")
fin.write("impsite = ./dump/impsite.txt\n")
fin.write("intfile = ./dump/intfile.txt\n")
fin.write("ibath = %s\n"%ibath)
fin.write("getrdm2 = yes\n")
fin.write("N = %d\n"%nemb)
fin.write("Npart = %d\n"%nelec_emb)
fin.write("Nimp = %d\n"%nimp)
fin.write("U = %f\n"%U)
fin.write("nsweeps = 8\n")
fin.write("sweeps\n")
fin.write("{\n")
fin.write("maxm  minm  cutoff  niter  noise\n")
fin.write("100   20    1e-12   2      1E-6\n")
fin.write("200   20    1e-12   2      1E-7\n")
fin.write("400   20    1e-12   2      0\n")
fin.write("800   20    1e-12   2      1e-11\n")
fin.write("800   20    1e-12   2      0\n")
fin.write("1000   20    1e-12   2      0\n")
fin.write("1200   20    1e-12   2      0\n")
fin.write("1600   20    1e-16   2      0\n")
fin.write("2000   20    1e-16   2      0\n")
fin.write("}\n")
fin.write("quiet = yes\n")
fin.write("}\n")
fin.close()

# run mps
subprocess.call(["../impsolver_dmet", "input_solver"])

# fci reference on solving impurity
h1fci = (h1emb[0], h1emb[1])
g2fci = (Vemb*0.0, Vemb, Vemb*0.0)
e, c = fcisolver.kernel(h1fci, g2fci, nemb, nelec_emb)
ci0 = np.random.rand(c.shape[0], c.shape[1])
ci0 /= np.linalg.norm(ci0)
#e, c = fcisolver.kernel(h1fci, g2fci, nemb, nelec_emb, ci0=ci0)
rdm1s,rdm2s = fcisolver.make_rdm12s(c, nemb, nelec_emb)
rdm1s = np.asarray(rdm1s).reshape(2*nemb,nemb)
rdm2ab = rdm2s[1]
print "FCI energy: ", e

if ibath:
    # test the difference of rdms
    emps = np.loadtxt("./dump/energy.txt")
    print "Energy difference: ", emps - e

    rdm1mps = np.loadtxt("./dump/rdm1s.txt")
    print "RDM1 difference: ", np.linalg.norm(rdm1mps-rdm1s)
    rdm2mps = np.loadtxt("./dump/rdm2.txt")
    rdm2mps = rdm2mps.reshape(nemb,nemb,nemb,nemb)
    print "RDM2 difference: ", np.linalg.norm(rdm2ab-rdm2mps)
else:
    emps = np.loadtxt("./dump/energy.txt")
    print "Energy difference: ", emps - e

    rdm1mps = np.loadtxt("./dump/rdm1s.txt")
    print "RDM1 difference: ", np.linalg.norm(rdm1mps-rdm1s)

    rdm2mps = np.loadtxt("./dump/rdm2.txt")
    rdm2fci_diag = np.zeros(nemb)
    for i in range(nemb):
        rdm2fci_diag[i] = rdm2ab[i,i,i,i]
    print "RDM2 difference: ", np.linalg.norm(rdm2mps-rdm2fci_diag)



    

