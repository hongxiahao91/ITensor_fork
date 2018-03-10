#include "itensor/all.h"

using namespace itensor;

int main(){

    // define Hubbard model Hamiltonian
    int N = 40;
    int Npart = 40;
    auto U = 1.0;
    auto t = 1.0;
    //auto sites = Hubbard(N);
    //auto ampo = AutoMPO(sites);

    auto sites = Hubbard(N);
    auto ampo = AutoMPO(sites);
    for(int i = 1; i <= N; ++i)
        {
        ampo += U,"Nupdn",i;
        }
    for(int b = 1; b < N; ++b)
        {
        ampo += -t,"Cdagup",b,"Cup",b+1;
        ampo += -t,"Cdagup",b+1,"Cup",b;
        ampo += -t,"Cdagdn",b,"Cdn",b+1;
        ampo += -t,"Cdagdn",b+1,"Cdn",b;
        }
    auto H = IQMPO(ampo);

    int p = Npart;
    auto state = InitState(sites);
    for(int i = N; i >= 1; --i) 
        {
        if(p > i)
            {
            println("Doubly occupying site ",i);
            state.set(i,"UpDn");
            p -= 2;
            }
        else
        if(p > 0)
            {
            println("Singly occupying site ",i);
            state.set(i,(i%2==1 ? "Up" : "Dn"));
            p -= 1;
            }
        else
            {
            state.set(i,"Emp");
            }
        }


    //for(int i=1;i<=N;++i){
    //    ampo += U, "Nupdn", i;

    //}

    //for(int b=1; b<N; ++b){
    //    ampo += -t,"Cdagup",b, "Cup", b+1;
    //    ampo += -t,"Cdagup",b+1, "Cup", b;
    //    ampo += -t,"Cdagdn",b, "Cdn", b+1;
    //    ampo += -t,"Cdagdn",b+1, "Cdn", b;
    //}

    //auto H = IQMPO(ampo);
    auto psi = IQMPS(state);

    auto sweeps = Sweeps(5);
    sweeps.maxm() = 10, 20, 40, 80, 160;
    sweeps.cutoff() = 1E-10;
    auto energy = dmrg(psi, H, sweeps);
    
    printfln("Ground state energy = %.10f", energy/N);
    return 0; 
   
}
