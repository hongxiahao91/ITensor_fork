#include "itensor/all.h"
#include <iostream>
#include <string>
#include <fstream>

using namespace itensor;
using namespace std;


double*** get_rdm1s(IQMPS psi, int N);
double* get_rdm2diag(IQMPS psi, int N);
double**** get_rdm2s(IQMPS psi, int N);

//**********************************************************************************
int main(int argc, char* argv[])
    {
    // read input
    auto input   = InputGroup(argv[1],"input");

    auto N       = input.getInt("N");
    auto Npart   = input.getInt("Npart",N); //# of particles, default is N (half filling)
    auto nimp    = input.getInt("Nimp",0);
    auto U       = input.getReal("U",0);
    auto ibath   = input.getYesNo("ibath",false);
    auto getrdm2 = input.getYesNo("getrdm2",false);
    auto hamfile = input.getString("hamfile");
    auto intfile = input.getString("intfile");
    auto impid   = input.getString("impsite");
    auto outdir  = input.getString("outdir");

    auto nsweeps = input.getInt("nsweeps");
    auto quiet   = input.getYesNo("quiet",false);

    auto table   = InputGroup(input,"sweeps");
    auto sweeps  = Sweeps(nsweeps,table);

    //println(sweeps);
    //println("Hamiltonian file: ",hamfile);

    //
    // Initialize the site degrees of freedom.
    //
    auto sites = Hubbard(N, {"ConserveNf",true});
    //
    // Create the Hamiltonian using AutoMPO
    //
    auto ampo = AutoMPO(sites);
    
    // two body term
    if(ibath)
        {
        double v;
        ifstream vfile(intfile);
        for(int i=1;i<=N;++i)
        for(int j=1;j<=N;++j)
        for(int k=1;k<=N;++k)
        for(int l=1;l<=N;++l)
            {
            vfile >> v;
            if(i==j && k==l)
                ampo += v, "Nup", i, "Ndn", k;
            else if(i==j && k!=l)
                ampo += v, "Nup", i, "Cdagdn",k,"Cdn",l;
            else if(k==l && i!=j)
                ampo += v, "Cdagup",i,"Cup",j,"Ndn",k;
            else
                ampo += v, "Cdagup",i,"Cup",j,"Cdagdn",k,"Cdn",l;
            }
        }
    else
        {
        ifstream impf(impid);
        int imp_id;
        for(int i=1; i<=nimp; ++i) 
            {
            impf >> imp_id;
            ampo += U,"Nupdn",imp_id;
            }
        }
    //cout << ampo << endl;
    // input one-body Hamiltonian from input
    ifstream hfile(hamfile);
    double htemp=0;
    // spin up
    for (int i=1; i<=N; ++i)
    for (int j=1; j<=N; ++j)
        {
        hfile >> htemp;
        if(i==j) ampo += htemp, "Nup", i;
        else     ampo += htemp, "Cdagup",i,"Cup",j;
        }
    // spin down
    for (int i=1; i<=N; ++i)
    for (int j=1; j<=N; ++j)
        {
        hfile >> htemp;
        if(i==j) ampo += htemp, "Ndn", i;
        else     ampo += htemp, "Cdagdn",i,"Cdn",j;
        }


    auto H = IQMPO(ampo);

    //
    // Set the initial wavefunction matrix product state
    // to be a Neel state.
    //
    auto state = InitState(sites);
    int p = Npart;
    for(int i = N; i >= 1; --i) 
        {
        if(p > i)
            {
            //println("Doubly occupying site ",i);
            state.set(i,"UpDn");
            p -= 2;
            }
        else
        if(p > 0)
            {
            //println("Singly occupying site ",i);
            state.set(i,(i%2==1 ? "Up" : "Dn"));
            p -= 1;
            }
        else
            {
            state.set(i,"Emp");
            }
        }

    auto psi = IQMPS(state);

    //Print(totalQN(psi));
    //
    // DMRG calculation
    
    //time_t ti_dmrg = time(0);

    auto energy = dmrg(psi,H,sweeps,{"Quiet",quiet, "PrintEigs", false});

    //time_t tf_dmrg = time(0);
    //cout << "Time used to do DMRG is: " << tf_dmrg-ti_dmrg << endl;
    //printfln("\nDMRG Energy = %.10f",energy);


    ofstream enef(outdir + "energy.txt");
    enef << format("%.14f  \n", energy);
    enef.close();
    
    auto*** rdm1s = get_rdm1s(psi, N);
    ofstream rdm1f(outdir + "rdm1s.txt");
    for(int s=0; s<2; ++s)
        for(int i=0; i<N; ++i)
            {
            for(int j=0; j<N; ++j)
                rdm1f << format("%.14f  ",rdm1s[s][i][j]);
            rdm1f << "\n";
            }
    rdm1f.close();

    if(getrdm2)
    {
        ofstream rdm2f(outdir + "rdm2.txt");
        if(ibath)
            {
            //time_t ti_rdm2 = time(0);
            auto****rdm2s = get_rdm2s(psi,N);
            //time_t tf_rdm2 = time(0);
            //cout << "Time used to calculate RDM2 is: " << tf_rdm2-ti_rdm2 << endl;
            for(int i=0;i<N;++i)
            for(int j=0;j<N;++j)
            for(int k=0;k<N;++k)
                {
                for(int l=0;l<N;++l)
                    rdm2f << format("%.14f  ", rdm2s[i][j][k][l]);
                rdm2f << "\n";
                }
            }
        else
            {
            auto*rdm2 = get_rdm2diag(psi, N);
            for(int i=0;i<N;++i)
                rdm2f << format("%.14f  ", rdm2[i]); 
            }
        rdm2f.close();
    }

    return 0;
    }

//**********************************************************************************
double *get_rdm2diag(IQMPS psi, int N)
    {
    auto sites = psi.sites();
    double* rdm2 = 0;
    rdm2 = new double[N];
    for(int i=1; i<=N; ++i)
        {
        int ind = i;
        psi.position(ind);
        auto res = psi.A(ind)*sites.op("Nupdn", ind)*dag(prime(psi.A(ind), Site));
        rdm2[i-1] = res.real();
        }

    return rdm2;
    }

//**********************************************************************************
double ****get_rdm2s(IQMPS psi, int N)
    {
    //TODO use Jordan-Wigner string to accelerate the evaluation of 2RDM
    // A brute-force way to calculate 2RDM
    auto sites = psi.sites();
    // allocating the matrix
    double ****rdm2 = 0;
    rdm2 = new double***[N];
    for(int i=0;i<N;++i)
        {
        rdm2[i] = new double**[N];
        for (int j=0;j<N;++j)
            {
            rdm2[i][j] = new double *[N];
            for (int k=0;k<N;++k)
                rdm2[i][j][k] = new double[N];
            }
        }
    // off-diagonal terms
    for(int i=1;  i<=N;++i)
    for(int j=i+1;j<=N;++j)
    for(int k=1;  k<=N;++k)
    for(int l=k+1;l<=N;++l)
        {
        // (i-j)(k-l) > 0
        auto ampo = AutoMPO(sites);
        ampo += 1.0, "Cdagup",i,"Cup",j,"Cdagdn",k,"Cdn",l;
        auto dop = IQMPO(ampo); 
        auto dijkl = overlap(psi,dop,psi);
        rdm2[i-1][j-1][k-1][l-1] = dijkl;
        rdm2[j-1][i-1][l-1][k-1] = dijkl;
        // (i-j)(k-l) < 0 here we exchanged l and k
        ampo = AutoMPO(sites);
        ampo += 1.0, "Cdagup",i,"Cup",j,"Cdagdn",l,"Cdn",k;
        dop = IQMPO(ampo);
        dijkl = overlap(psi,dop,psi);
        rdm2[i-1][j-1][l-1][k-1] = dijkl;
        rdm2[j-1][i-1][k-1][l-1] = dijkl;

        }
    //for(int i=1;  i<=N;  ++i)
    //for(int j=1;  j<=i-1;++j)
    //for(int k=1;  k<=N;  ++k)
    //for(int l=k+1;l<=N;  ++l)
    //    {
    //    auto ampo = AutoMPO(sites);
    //    ampo += 1.0, "Cdagup",i,"Cup",j,"Cdagdn",k,"Cdn",l;
    //    auto dop = IQMPO(ampo); 
    //    //auto psi_n = exactApplyMPO(dop,psi);
    //    //auto dijkl = overlap(psi,psi_n);
    //    auto dijkl = overlap(psi,dop,psi);
    //    rdm2[i-1][j-1][k-1][l-1] = dijkl;
    //    rdm2[j-1][i-1][l-1][k-1] = dijkl;
    //    }

    // partly diagonal terms
    for(int i=1; i<=N; ++i) 
    for(int k=1; k<=N; ++k) 
    for(int l=k+1; l<=N; ++l)
        {
        // i=j
        auto ampo = AutoMPO(sites);
        ampo += 1.0, "Nup", i, "Cdagdn", k, "Cdn", l;
        auto dop = IQMPO(ampo);
        auto dijkl = overlap(psi,dop,psi);
        rdm2[i-1][i-1][k-1][l-1] = dijkl;
        rdm2[i-1][i-1][l-1][k-1] = dijkl;
        // k=l
        ampo = AutoMPO(sites);
        ampo += 1.0, "Cdagup",k,"Cup",l, "Ndn", i;
        dop = IQMPO(ampo);
        dijkl = overlap(psi,dop,psi);
        rdm2[k-1][l-1][i-1][i-1] = dijkl;
        rdm2[l-1][k-1][i-1][i-1] = dijkl;
        }


    // diagonal terms
    for(int i=1;i<=N;++i)
    for(int k=1;k<=N;++k)
        {
        auto ampo = AutoMPO(sites);
        ampo += 1.0, "Nup", i, "Ndn", k;
        auto dop   = IQMPO(ampo);
        rdm2[i-1][i-1][k-1][k-1] = overlap(psi,dop,psi);
        //int ind = i;
        //psi.position(ind);
        //auto res = psi.A(ind)*sites.op("Nupdn", ind)*dag(prime(psi.A(ind), Site));
        //rdm2[i-1][i-1][i-1][i-1] = res;
        }
    
    return rdm2;
    }

//**********************************************************************************
double*** get_rdm1s(IQMPS psi, int N)
    {
    // N is the number of sites
    // return a (2,N,N) array with to store the 1RDM. 2 accounts for spin freedom
    // initialization
    auto sites = psi.sites();
    double*** rdm1 = 0;
    rdm1 = new double**[2];
    for(int i=0; i<2; ++i)
        {
        rdm1[i] = new double*[N];
        for(int j=0; j<N; ++j)
            rdm1[i][j] = new double[N];
        }
    
    // off-diagonal terms
    int lind, rind, k;
    for(int i=1; i<N; ++i)
        {
        lind = i;
        auto AdagupF_i = sites.op("Adagup*F", lind);
        auto Adagdn_i = sites.op("Adagdn", lind);
        psi.position(lind);
        auto ir = commonIndex(psi.A(lind), psi.A(lind+1), Link);
        auto Corrup = psi.A(lind)*AdagupF_i*dag(prime(psi.A(lind),Site,ir));
        auto Corrdn = psi.A(lind)*Adagdn_i*dag(prime(psi.A(lind),Site,ir));
        for(int j=i+1; j<=N; ++j)
            {
            rind = j;
            auto Aup_j = sites.op("Aup", rind);
            auto AdnF_j = sites.op("F*Adn", rind);
            //measure the correlation function
            auto Corrupij = Corrup * psi.A(rind);
            Corrupij *= Aup_j;
            auto Corrdnij = Corrdn * psi.A(rind);
            Corrdnij *= AdnF_j;
            auto jl = commonIndex(psi.A(rind), psi.A(rind-1), Link);
            Corrupij *= dag(prime(psi.A(rind),jl,Site));
            Corrdnij *= dag(prime(psi.A(rind),jl,Site));
                
            rdm1[0][i-1][j-1] = Corrupij.real();
            rdm1[0][j-1][i-1] = Corrupij.real();
            rdm1[1][i-1][j-1] = Corrdnij.real();
            rdm1[1][j-1][i-1] = Corrdnij.real();
            // apply F to the rind site
            k = rind;
            Corrup *= psi.A(k);
            Corrup *= sites.op("F", k);
            Corrup *= dag(prime(psi.A(k)));
            Corrdn *= psi.A(k);
            Corrdn *= sites.op("F", k);
            Corrdn *= dag(prime(psi.A(k)));
            }
        }
    
    // diagonal terms
    for (int i=1; i<=N; ++i)
        {
        int ind = i;
        psi.position(ind);

        auto resup = psi.A(ind)*sites.op("Nup", ind)*dag(prime(psi.A(ind), Site));
        auto resdn = psi.A(ind)*sites.op("Ndn", ind)*dag(prime(psi.A(ind), Site));
        rdm1[0][i-1][i-1] = resup.real();
        rdm1[1][i-1][i-1] = resdn.real();
        }
    return rdm1;
    }


