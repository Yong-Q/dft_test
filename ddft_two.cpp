// Two-field conserved DDFT + space-charge for unidirectional Li dendrites.
//   n: solid Li density (conserved Model B, spectral 2D-FFT CH -- mass-exact)
//   c: Li+ cation density (conserved Model B, Fickian + migration)
//   phi: potential (Poisson, space charge; solid equipotential)
//
//   d n/dt = M_n lap[ f'(n) - kappa lap n ] + R_BV/Cm   (conserved Model B DDFT)
//   d c/dt = D lap c + M_mig div(c grad phi) - R_BV
//   lap phi = -beta (c - 1)  (space charge; beta = RT/(lamD^2 F))
//   R_BV = R0 |grad n| [ c exp(-a eta) - exp((1-a)eta) ]   (interface-area rate)
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <fftw3.h>
#include <chrono>
#include <algorithm>
#include <numeric>
using namespace std;

static inline double clmp(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }
static inline double Hfun(double n){ n=clmp(n,0.0,1.0); return n*n*n*(6*n*n-15*n+10); }
static inline double fprime(double n,double W){ return 2.0*W*n*(1.0-n)*(1.0-2.0*n); }

struct FFTDiff {
    int nx,ny; double dx,dy; fftw_plan fwd,bwd; vector<double> in,outbuf; fftw_complex* out;
    vector<double> dia,lo,up,r,sol;
    FFTDiff(int nx_,int ny_,double dx_,double dy_):nx(nx_),ny(ny_),dx(dx_),dy(dy_){
        in.resize(nx*ny); outbuf.resize((size_t)nx*(ny/2+1)*2); out=(fftw_complex*)outbuf.data();
        int n[]={ny}; int hm=nx, id=ny, od=ny/2+1;
        fwd=fftw_plan_many_dft_r2c(1,n,hm,in.data(),NULL,1,id,out,NULL,1,od,FFTW_ESTIMATE);
        bwd=fftw_plan_many_dft_c2r(1,n,hm,out,NULL,1,od,in.data(),NULL,1,id,FFTW_ESTIMATE);
        dia.resize(nx); lo.resize(nx); up.resize(nx); r.resize(nx); sol.resize(nx);
    }
    ~FFTDiff(){ fftw_destroy_plan(fwd); fftw_destroy_plan(bwd); }
    void solve(vector<double>&u,const vector<double>&rhs,double a,double b,double bc0,double bc1,bool farNoFlux=false){
        int N=nx*ny;
        for(int i=0;i<N;++i) in[i]=rhs[i];
        fftw_execute(fwd);
        for(int k=0;k<ny/2+1;++k){
            double ky=2*M_PI*k/(ny*dy), ky2=ky*ky;
            double d=a+b*ky2+2*b/(dx*dx), o=-b/(dx*dx);
            for(int i=0;i<nx;++i){ dia[i]=d; lo[i]=(i>0)?o:0; up[i]=(i<nx-1)?o:0; r[i]=out[i*(ny/2+1)+k][0]; }
            double b0=(k==0)?bc0*ny:0.0, b1=(k==0)?bc1*ny:0.0;
            dia[0]=1; up[0]=0; r[0]=b0;
            if(farNoFlux){ dia[nx-1]=d; lo[nx-1]=-2.0*o; up[nx-1]=0; r[nx-1]=out[(nx-1)*(ny/2+1)+k][0]; }
            else { dia[nx-1]=1; lo[nx-1]=0; r[nx-1]=b1; }
            for(int i=1;i<nx;++i){ double m=lo[i]/dia[i-1]; dia[i]-=m*up[i-1]; r[i]-=m*r[i-1]; }
            sol[nx-1]=r[nx-1]/dia[nx-1];
            for(int i=nx-2;i>=0;--i) sol[i]=(r[i]-up[i]*sol[i+1])/dia[i];
            for(int i=0;i<nx;++i){ out[i*(ny/2+1)+k][0]=sol[i]; out[i*(ny/2+1)+k][1]=0; }
        }
        fftw_execute(bwd);
        for(int i=0;i<N;++i) u[i]=in[i]/ny;
    }
};

int main(int argc,char**argv){
    int nx=600,ny=256,nsteps=20000;
    double dx=0.2,dy=0.2,dt=0.005;
    double D=100.0, W=1.11, kappa=0.80, Mn=0.05;
    double kappa_num=0.02;  // weak numerical gradient (interface stability only); physical surface tension is in d0 (Gibbs-Thomson)
    double i0=1.0, c0=1000.0, Cm=76.9, alpha=0.5;
    double eta_volt=-0.45, phi_app=1.0, lamD=1.0;
    int nphi=12;
    double x0=20.0, initw=0.4;
    double Rfac=1.0;
    double d0=0.223;  // capillary length (um) = gamma/(RT c0)  [Gibbs-Thomson]
    double d_stress=2.7; // (kept for reference; deposition-generated stress uses A below)
    double Astr=17.9;   // stress coeff = E_eff Vm/(RT), deposition-generated
    double tau_s=0.05;  // stress relaxation time (s), Li soft/fast
    int seed=1, modelA=1;
    int finitemode=0; double cinit=1.0;
    int whisker=0, wyc=-1, nch=1; double ww=0.0, wh0=0.0;   // whisker-channel mode (nch periodic channels)
    int jd=0; double jd_scale=2000.0;
    int rootmode=0; double rootH=4.0;
    int wallmode=0;
    double Ms=0.0, psi_c=0.0, Dpsi=10.0, a_str=1.0;
    int ccdep=0;
    double eps4=0.0, eps6=0.0;   // 4- and 6-fold anisotropy
    double bumpA=0.0, bumpw=2.0;  // single localized seed bump (symmetry breaking)
    int nucleus=0; double nucR=3.0;  // point nucleus seed (no film)
    double tnoise=0.0;            // time-dependent interface noise (seeds side-branching)
    int mrho=0; double mfloor=0.1;   // density-dependent mobility M(rho)=Mn*(mfloor+rho)
    double Pstack=0.0, Hhard=10.0; vector<double> contact;   // current-density (field-concentration) driven deposition
    int seihetero=0; double sei_s=1.0, sei_corr=1.0;   // SEI heterogeneity: i0(y)=i0*exp(s*G(y))
    double roughA=0.0, roughL=0.5;                     // measured surface roughness RMS (um) & corr length (um)
    vector<double> i0f;
    for(int a=1;a<argc;++a){
        if(!strcmp(argv[a],"-nx")) nx=atoi(argv[++a]);
        else if(!strcmp(argv[a],"-ny")) ny=atoi(argv[++a]);
        else if(!strcmp(argv[a],"-n")) nsteps=atoi(argv[++a]);
        else if(!strcmp(argv[a],"-dx")){dx=atof(argv[++a]);dy=dx;}
        else if(!strcmp(argv[a],"-dt")) dt=atof(argv[++a]);
        else if(!strcmp(argv[a],"-D")) D=atof(argv[++a]);
        else if(!strcmp(argv[a],"-W")) W=atof(argv[++a]);
        else if(!strcmp(argv[a],"-kappa")) kappa=atof(argv[++a]);
        else if(!strcmp(argv[a],"-eta")) eta_volt=atof(argv[++a]);
        else if(!strcmp(argv[a],"-i0")) i0=atof(argv[++a]);
        else if(!strcmp(argv[a],"-Cm")) Cm=atof(argv[++a]);
        else if(!strcmp(argv[a],"-Mn")) Mn=atof(argv[++a]);
        else if(!strcmp(argv[a],"-Rfac")) Rfac=atof(argv[++a]);
        else if(!strcmp(argv[a],"-d0")) d0=atof(argv[++a]);
        else if(!strcmp(argv[a],"-nphi")) nphi=atoi(argv[++a]);
        else if(!strcmp(argv[a],"-kn")) kappa_num=atof(argv[++a]);
        else if(!strcmp(argv[a],"-dst")) d_stress=atof(argv[++a]);
        else if(!strcmp(argv[a],"-Astr")) Astr=atof(argv[++a]);
        else if(!strcmp(argv[a],"-tau")) tau_s=atof(argv[++a]);
        else if(!strcmp(argv[a],"-phi")) phi_app=atof(argv[++a]);
        else if(!strcmp(argv[a],"-lamD")) lamD=atof(argv[++a]);
        else if(!strcmp(argv[a],"-seed")) seed=atoi(argv[++a]);
        else if(!strcmp(argv[a],"-modelA")) modelA=atoi(argv[++a]);
        else if(!strcmp(argv[a],"-finite")) finitemode=1;
        else if(!strcmp(argv[a],"-cinit")) cinit=atof(argv[++a]);
        else if(!strcmp(argv[a],"-whisker")){ whisker=1; ww=atof(argv[++a]); wh0=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-root")){ rootmode=1; rootH=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-wall")) wallmode=1;
        else if(!strcmp(argv[a],"-nch")) nch=atoi(argv[++a]);
        else if(!strcmp(argv[a],"-jd")){ jd=1; jd_scale=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-sei")){ seihetero=1; sei_s=atof(argv[++a]); sei_corr=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-rough")){ roughA=atof(argv[++a]); roughL=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-stress")){ Ms=atof(argv[++a]); psi_c=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-ccdep")) ccdep=1;
        else if(!strcmp(argv[a],"-eps")) eps4=atof(argv[++a]);
        else if(!strcmp(argv[a],"-eps6")) eps6=atof(argv[++a]);
        else if(!strcmp(argv[a],"-bump")){ bumpA=atof(argv[++a]); bumpw=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-nucleus")){ nucleus=1; nucR=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-tnoise")) tnoise=atof(argv[++a]);
        else if(!strcmp(argv[a],"-mrho")){ mrho=1; mfloor=atof(argv[++a]); }
        else if(!strcmp(argv[a],"-pressure")){ Pstack=atof(argv[++a]); Hhard=atof(argv[++a]); }
    }
    double RT=8.314*300.0, F=96485.0;
    double eta=eta_volt*F/RT;
    double k=(i0*10.0)/(F*c0)*1e6;
    double delta=sqrt(kappa/W);
    double R0=Rfac*k/delta;               // = i0/(F c0 delta): reaction prefactor; Δn carries 1/Cm
    double efa=exp(-alpha*eta), efb=exp((1-alpha)*eta);
    double Mmig=0.0;  // ambipolar (equal-D salt): migration cancels -> purely diffusive
    double lam=2.0*W;
    double sig_s=100.0, sig_l=1.0;      // conductivities (metal >> electrolyte); ratio sets field crowding
    fprintf(stderr,"eta=%.2f R0=%.4f Mmig=%.1f sig_s/sig_l=%.0f\n",eta,R0,Mmig,sig_s/sig_l);

    int N=nx*ny, nyh=ny/2+1;
    vector<double> psi(N,0.0);
    vector<double> n(N), c(N), phi(N), dN(N), rhs(N), cbuf(N), curv(N), dep(N,0.0), nold(N), dndt(N), n_init(N);
    vector<double> cm(N,1.0);   // anion (PF6-) field for space charge
    // three complex buffers for spectral CH
    vector<double> bufA((size_t)nx*nyh*2), bufB((size_t)nx*nyh*2), bufC((size_t)nx*nyh*2);
    fftw_complex* A=(fftw_complex*)bufA.data();
    fftw_complex* B=(fftw_complex*)bufB.data();
    fftw_complex* C=(fftw_complex*)bufC.data();
    fftw_plan pA=fftw_plan_dft_r2c_2d(nx,ny,rhs.data(),A,FFTW_ESTIMATE);
    fftw_plan pB=fftw_plan_dft_r2c_2d(nx,ny,rhs.data(),B,FFTW_ESTIMATE);
    fftw_plan pC=fftw_plan_dft_r2c_2d(nx,ny,rhs.data(),C,FFTW_ESTIMATE);
    fftw_plan inv=fftw_plan_dft_c2r_2d(nx,ny,A,rhs.data(),FFTW_ESTIMATE);
    FFTDiff helm(nx,ny,dx,dy);

    srand(seed);
    double Ly=ny*dy;
    // --- SEI heterogeneity: local exchange current i0(y)=i0*exp(s*G(y)), G correlated Gaussian (corr len sei_corr) ---
    i0f.assign(ny,1.0);
    if(seihetero){
        vector<double> g(ny,0.0);
        double kc=2*M_PI/sei_corr;
        srand(seed+101);
        for(int m=0;m<48;++m){
            double km=kc*(0.5+2.0*m/48.0);
            double ph=2*M_PI*((double)rand()/RAND_MAX);
            double am=1.0/sqrt(1.0+km*km*sei_corr*sei_corr*0.1);
            for(int j=0;j<ny;++j) g[j]+=am*cos(km*j*dy+ph);
        }
        double m0=0,s0=0; for(int j=0;j<ny;++j)m0+=g[j]; m0/=ny;
        for(int j=0;j<ny;++j)s0+=(g[j]-m0)*(g[j]-m0); s0=sqrt(s0/ny)+1e-12;
        for(int j=0;j<ny;++j) i0f[j]=exp(sei_s*(g[j]-m0)/s0);
        fprintf(stderr,"SEI hetero: s=%.2f corr=%.2f um  i0 range[%.2f,%.2f]\n",sei_s,sei_corr,i0f.empty()?0:*min_element(i0f.begin(),i0f.end()),*max_element(i0f.begin(),i0f.end()));
    }
    // --- stack-pressure contact (Greenwood-Williamson): real contact fraction = P/H ---
    //   The "defect" that seeds growth is a contact asperity; its number is set by applied pressure.
    contact.assign(ny,1.0);
    if(Pstack>0){
        vector<double> G(ny); srand(seed+55);
        for(int j=0;j<ny;++j){ double g=0; for(int q=0;q<6;++q) g+=((double)rand()/RAND_MAX); G[j]=g; }  // ~Gaussian asperity heights
        vector<int> idx(ny); iota(idx.begin(),idx.end(),0);
        sort(idx.begin(),idx.end(),[&](int a,int b){return G[a]<G[b];});
        double frac=clmp(Pstack/Hhard,0.0,1.0); int ncon=(int)(frac*ny+0.5); if(ncon<1)ncon=1;
        for(int j=0;j<ny;++j) contact[j]=0.0;
        for(int k=0;k<ncon;++k) contact[idx[ny-1-k]]=1.0;   // top ncon asperities in contact
        fprintf(stderr,"stack pressure P/H=%.2f -> contact fraction=%.2f (%d/%d)\n",Pstack/Hhard,frac,ncon,ny);
    }
    int wyc_grid = (whisker && wyc<0) ? ny/2 : wyc;
    for(int i=0;i<nx;++i){ double x=i*dx;
        for(int j=0;j<ny;++j){ double y=j*dy;
            double v;
            if(nucleus){
                double yc=0.5*Ly; double r=sqrt((x-4.0*dx)*(x-4.0*dx)+(y-yc)*(y-yc));
                v=(r<nucR)?1.0:0.0;
            } else if(whisker){
                // solid film (x<x0) everywhere + nch narrow columns of half-width ww/2 rising to x0+wh0
                v=0.0;
                if(x<x0) v=0.5*(1.0-tanh((x-x0)/initw));
                else {
                    double ypos=fmod(y, Ly/nch);
                    double dyj=fabs(ypos-0.5*Ly/nch);
                    if(dyj<0.5*ww && x<x0+wh0) v=1.0;
                }
            } else {
                double sh;
                if(bumpA>0){
                    double yc=0.5*Ly;
                    sh=bumpA*exp(-0.5*pow((y-yc)/bumpw,2.0));   // single localized bump
                } else if(roughA>0){
                    // measured surface roughness: correlated random field, RMS amplitude roughA (um), corr len roughL (um)
                    sh=0.0;
                    for(int m=0;m<24;++m){
                        double km=2*M_PI*(0.5+m)/roughL;
                        double ph=2*M_PI*((double)rand()/RAND_MAX);
                        double am=1.0/sqrt(1.0+km*km*roughL*roughL*0.05);
                        sh+=am*cos(km*y+ph);
                    }
                    sh*=roughA/1.6;   // rough scale to ~ roughA RMS (approx)
                } else {
                    sh=0.6*sin(2*M_PI*y/10.0)+0.3*sin(2*M_PI*y/20.0+1.0)
                       +0.1*((double)rand()/RAND_MAX*2-1);
                }
                v=0.5*(1.0-tanh((x-x0-sh)/initw));
            }
            n[i*ny+j]=v;
            c[i*ny+j]=cinit*(1.0-Hfun(v));
        } }
    for(int j=0;j<ny;++j){ c[j]=0.0; c[(nx-1)*ny+j]=1.0; }
    n_init=n;   // save initial (smooth) profile for rigid SEI-wall freeze
    for(int i=0;i<nx;++i){ double x=i*dx; for(int j=0;j<ny;++j) phi[i*ny+j]=phi_app*(x/(nx*dx)); }
    for(int j=0;j<ny;++j){ phi[j]=0.0; phi[(nx-1)*ny+j]=phi_app; }

    auto t0=chrono::high_resolution_clock::now();
    int snap=0;
    for(int step=0;step<=nsteps;++step){
        // ---- potential: variable-conductivity Laplace  div(sigma(n) grad phi) = 0 ----
        //   sigma(n) = sigma_l + (sigma_s - sigma_l) h(n)   (metal >> electrolyte)
        //   Dirichlet phi = eta_volt at electrode (x=0), phi = 0 at counter (x=nx-1).
        //   The solid is a good conductor; the field crowds the electrolyte at protrusions
        //   -> local overpotential eta=phi is larger at tips -> tip growth (HV/Cogswell mechanism).
        //   Red-black SOR, updated every step (quasi-static).
        for(int it=0;it<nphi;++it){
            for(int par=0;par<2;++par){
            #pragma omp parallel for
            for(int i=1;i<nx-1;++i) for(int j=0;j<ny;++j){
                if(((i+j)&1)!=par) continue;
                int jm=(j-1+ny)%ny, jp=(j+1)%ny, id=i*ny+j;
                double hn=Hfun(n[id]);
                // electrolyte conductivity proportional to salt concentration (physical: sigma = F z mu c)
                double cl0 = ccdep ? 0.02 : 1.0;
                double slv  = sig_l*(cl0 + (1.0-cl0)*clmp(c[id],0.0,1.0));
                double sc=slv+(sig_s-sig_l)*hn;
                double hE=Hfun(n[(i+1)*ny+j]), hW=Hfun(n[(i-1)*ny+j]);
                double hN=Hfun(n[i*ny+jp]),     hS=Hfun(n[i*ny+jm]);
                double slvE=sig_l*(cl0+(1.0-cl0)*clmp(c[(i+1)*ny+j],0,1));
                double slvW=sig_l*(cl0+(1.0-cl0)*clmp(c[(i-1)*ny+j],0,1));
                double slvN=sig_l*(cl0+(1.0-cl0)*clmp(c[i*ny+jp],0,1));
                double slvS=sig_l*(cl0+(1.0-cl0)*clmp(c[i*ny+jm],0,1));
                double sE=0.5*(sc+slvE+(sig_s-sig_l)*hE);
                double sW=0.5*(sc+slvW+(sig_s-sig_l)*hW);
                double sN=0.5*(sc+slvN+(sig_s-sig_l)*hN);
                double sS=0.5*(sc+slvS+(sig_s-sig_l)*hS);
                double den=(sE+sW+sN+sS);
                double val=(sE*phi[(i+1)*ny+j]+sW*phi[(i-1)*ny+j]+sN*phi[i*ny+jp]+sS*phi[i*ny+jm])/den;
                phi[id]+=1.7*(val-phi[id]);     // SOR
            }
            }
            for(int j=0;j<ny;++j){ phi[j]=eta_volt; phi[(nx-1)*ny+j]=0.0; }
        }
        // ---- interface curvature kappa = div( grad n / |grad n| ) ----
        #pragma omp parallel for
        for(int i=1;i<nx-1;++i) for(int j=0;j<ny;++j){
            int jm=(j-1+ny)%ny, jp=(j+1)%ny, id=i*ny+j;
            double gx=(n[(i+1)*ny+j]-n[(i-1)*ny+j])/(2*dx);
            double gy=(n[i*ny+jp]-n[i*ny+jm])/(2*dy);
            double gm=sqrt(gx*gx+gy*gy)+1e-9;
            double nxm=(n[i*ny+j]-n[(i-1)*ny+j])/dx, nxp=(n[(i+1)*ny+j]-n[i*ny+j])/dx;
            double nym=(n[i*ny+j]-n[i*ny+jm])/dy, nyp=(n[i*ny+jp]-n[i*ny+j])/dy;
            double gmp=sqrt(nxp*nxp+nyp*nyp)+1e-9, gmm=sqrt(nxm*nxm+nym*nym)+1e-9;
            curv[id]=(nxp/gmp - nxm/gmm)/dx + (nyp/gmp - nym/gmm)/dy;
        }
        // ---- stress concentration at protrusions (Wang/Dasgupta whisker model) ----
        //   The deposited film is under compression; stress concentrates at surface protrusions
        //   (shorter load path / grain-boundary wedge). psi(y) = smoothed protrusion height h(y)-<h>.
        //   Root extrusion is possible only where psi > psi_c (Li yield). Capillarity (d0) sets width.
        if(Ms>0){
            vector<double> xs(ny,0.0);
            for(int j=0;j<ny;++j){ xs[j]=x0; for(int i=1;i<nx;++i){ if(n[i*ny+j]<0.5){ xs[j]=i*dx; break; } } }
            double xm=0; for(int j=0;j<ny;++j) xm+=xs[j]; xm/=ny;

            // smooth over correlation length ~ roughL
            vector<double> hs(ny,0.0); int rad=(int)(max(roughL,2*dx)/dy)+1;
            for(int j=0;j<ny;++j){ double a=0,w=0; for(int m=-rad;m<=rad;++m){ int jj=(j+m+ny)%ny; double ww=exp(-0.5*(m*1.0/rad)*(m*1.0/rad)); a+=ww*xs[jj]; w+=ww; } hs[j]=a/w; }
            for(int i=0;i<N;++i){ int j=i%ny; psi[i]=hs[j]-xm; }   // psi = protrusion height above mean
        }

        // ---- reaction: Li+ (c) -> Li(s) (n) at the SOLID side of the interface only ----
        //   R = R0 |grad n| [ c exp(-a eta) - exp((1-a)eta) ],  eta = local overpotential + Gibbs-Thomson
        //   deposition only where n>0.5 (solid half) so the electrolyte (n~0) never nucleates spurious solid.
        #pragma omp parallel for
        for(int i=1;i<nx-1;++i) for(int j=0;j<ny;++j){
            int jm=(j-1+ny)%ny, jp=(j+1)%ny, id=i*ny+j;
            double nn=n[id];
            dN[id]=0.0;
            if(whisker){ double ypos=fmod(j*dy, Ly/nch); if(fabs(ypos-0.5*Ly/nch) >= 0.5*ww) continue; }   // reaction only inside channels
            if(nn<0.15 || nn>0.85) continue;           // deposit on the INTERFACE band only (liquid side advances the front; n=1 solid would be clamped/wasted; n<0.15 bulk electrolyte avoids spurious nucleation)
            if(rootmode && (i*dx) > x0+rootH) continue;  // root growth: deposit only at base
            if(nn>1.0) nn=1.0;
            double gx=(n[(i+1)*ny+j]-n[(i-1)*ny+j])/(2*dx);
            double gy=(n[i*ny+jp]-n[i*ny+jm])/(2*dy);
            double gm=sqrt(gx*gx+gy*gy)+1e-12;          // interface area density |grad n|
            if(jd){
                // ---- field-concentration (Mullins-Sekerka) deposition ----
                // local current density = sigma_l * (normal field across metal|electrolyte faces).
                // IMPORTANT: sample the ELECTROLYTE side (where the field is concentrated); the
                // solid interior is equipotential (E~0), so a central diff at the solid cell misses it.
                // Gibbs-Thomson penalty (d0*curv>0 sharp tip) suppresses grid-scale roughness -> finite wavelength.
                // local current = concentrated field peak in the electrolyte shell just OUTSIDE the tip
                // (diffuse interface smears the apex field ~2-3 cells out; scan outward and take the peak,
                //  which the converging current carries onto the tip -- grows with protrusion height -> positive feedback)
                double jloc=0.0;
                for(int k=1;k<=3;++k){
                    int ii=i+k; if(ii<nx-1&&n[ii*ny+j]<0.5){double e=fabs(phi[(ii+1)*ny+j]-phi[(ii-1)*ny+j])/(2*dx); if(e>jloc)jloc=e;}
                    ii=i-k;     if(ii>0   &&n[ii*ny+j]<0.5){double e=fabs(phi[(ii+1)*ny+j]-phi[(ii-1)*ny+j])/(2*dx); if(e>jloc)jloc=e;}
                }
                for(int k=1;k<=3;++k){
                    int jA=(j+k)%ny; if(n[i*ny+jA]<0.5){double e=fabs(phi[i*ny+((jA+1)%ny)]-phi[i*ny+((jA-1+ny)%ny)])/(2*dy); if(e>jloc)jloc=e;}
                    int jB=(j-k+ny)%ny; if(n[i*ny+jB]<0.5){double e=fabs(phi[i*ny+((jB+1)%ny)]-phi[i*ny+((jB-1+ny)%ny)])/(2*dy); if(e>jloc)jloc=e;}
                }
                if(jloc<1e-9) continue;                 // no electrolyte face here
                double gt=exp(-d0*curv[id]);            // capillary regularization
                if(gt>1.0) gt=1.0;
                double dn = jd_scale*jloc*gt*dt;        // solid fraction added per step
                double dcm = dn*Cm;                     // cation consumed (mass) / concentration drop
                if(c[id]>dcm){ dN[id]=dn; c[id]-=dcm; }
                else { dN[id]=c[id]/Cm; c[id]=0.0; }
                if(dN[id]<0)dN[id]=0;
            } else {
                double eta_local=96485.0*phi[id]/(8.314*300.0);
                double th=atan2(gy,gx); double d0e=d0*(1.0-eps4*cos(4.0*th));   // 4-fold anisotropic capillary
                double eta_eff=eta_local+d0e*curv[id];
                double efaL=exp(-alpha*eta_eff), efbL=exp((1-alpha)*eta_eff);
                double smul=1.0; if(Ms>0){ double dp=psi[id]-psi_c; if(dp>0) smul=1.0+Ms*dp; }
                double rc=R0*(seihetero?i0f[j]:1.0)*gm*smul*((Pstack>0)?contact[j]:1.0);     // BV x SEI i0(y) x stress-driven extrusion
                double cnew=(c[id]+dt*rc*efbL)/(1.0+dt*rc*efaL);
                if(cnew<0)cnew=0;
                dN[id]=(c[id]-cnew)/Cm;
                if(dN[id]<0)dN[id]=0;
                c[id]=cnew;
            }
        }
        // ---- n: conserved Model B (Cahn-Hilliard, Eyre convex-splitting), mass-exact ----
        //   d n/dt = Mn lap( f'(n) - kappa_num lap n ) + R   (conserved DDFT)
        for(int i=0;i<N;++i) nold[i]=n[i];
        for(int i=0;i<N;++i) rhs[i]=n[i];
        fftw_execute_dft_r2c(pA,rhs.data(),A);          // A = n_hat
        #pragma omp parallel for
        for(int i=0;i<N;++i){ double nn=clmp(n[i],0.0,1.0); rhs[i]=fprime(nn,W); }
        fftw_execute_dft_r2c(pB,rhs.data(),B);          // B = f'(n)_hat
        for(int i=0;i<N;++i) rhs[i]=dN[i];
        fftw_execute_dft_r2c(pC,rhs.data(),C);          // C = dN_hat
        double lamE=2.0*W;                              // Eyre convex-splitting stabilisation
        // n-dynamics: conserved DDFT (Model B) vs reaction-driven phase-field (Model A), selectable.
        //   Model B: dn/dt = Mn lap( f'(n) - kappa lap n ) + R  (strict conserved DDFT)
        //   Model A: dn/dt = -Mn f'(n) + R + small smoothing (phase-field electrodeposition, Hong-Viswanathan)
        for(int i=0;i<nx;++i) for(int j=0;j<nyh;++j){
            double kx=2*M_PI*((i<=nx/2)?i:(i-nx))/(nx*dx);
            double ky=2*M_PI*j/(ny*dy);
            double k2=kx*kx+ky*ky, k4=k2*k2;
            // anisotropic gradient-energy coefficient kappa(theta) = kappa*(1+eps4*cos4theta) (+eps6*cos6theta)
            double c4th=(k2>1e-12)?((kx*kx*kx*kx-6.0*kx*kx*ky*ky+ky*ky*ky*ky)/(k2*k2)):0.0;
            double c6th=(k2>1e-12)?((kx*kx*kx*kx*kx*kx-15.0*kx*kx*kx*kx*ky*ky+15.0*kx*kx*ky*ky*ky*ky-ky*ky*ky*ky*ky*ky)/(k2*k2*k2)):0.0;
            double kani=kappa_num*(1.0+eps4*c4th+eps6*c6th); if(kani<0.15*kappa_num)kani=0.15*kappa_num;
            int id=i*nyh+j;
            if(modelA){
                double den=1.0+dt*Mn*kani*k2;
                double re=(A[id][0]-dt*Mn*B[id][0]+C[id][0])/den;
                double im=(A[id][1]-dt*Mn*B[id][1]+C[id][1])/den;
                A[id][0]=re; A[id][1]=im;
            } else if(mrho){
                // variable mobility: use max mobility implicitly for stabilization, minimum (floor) for the k4 linear part
                double den=1.0+dt*Mn*(mfloor*lamE*k2+kani*k4);
                double re=(A[id][0]-dt*Mn*k2*B[id][0]+dt*Mn*mfloor*lamE*k2*A[id][0]+C[id][0])/den;
                double im=(A[id][1]-dt*Mn*k2*B[id][1]+dt*Mn*mfloor*lamE*k2*A[id][1]+C[id][1])/den;
                A[id][0]=re; A[id][1]=im;
            } else {
                double den=1.0+dt*Mn*(lamE*k2+kani*k4);
                double re=(A[id][0]-dt*Mn*k2*B[id][0]+dt*Mn*lamE*k2*A[id][0]+C[id][0])/den;
                double im=(A[id][1]-dt*Mn*k2*B[id][1]+dt*Mn*lamE*k2*A[id][1]+C[id][1])/den;
                A[id][0]=re; A[id][1]=im;
            }
        }
        fftw_execute_dft_c2r(inv,A,rhs.data());
        #pragma omp parallel for
        for(int i=0;i<N;++i) n[i]=clmp(rhs[i]/N,0.0,1.0);
        if(wallmode && whisker){ // rigid SEI walls: freeze n outside the channel to initial film/electrolyte
            int yc2=ny/2;
            for(int i=0;i<nx;++i)for(int j=0;j<ny;++j){
                double dyj=fabs((double)j-yc2)*dy;
                if(dyj>=0.5*ww){ n[i*ny+j]=n_init[i*ny+j]; }  // rigid SEI wall: freeze to initial smooth profile (stable)
            }
        }
        if(tnoise>0){   // thermal-like noise only at the diffuse interface (0.15<n<0.85)
            for(int i=0;i<N;++i){ double nn=n[i]; if(nn>0.15&&nn<0.85) n[i]=clmp(nn+tnoise*((double)rand()/RAND_MAX*2-1),0.0,1.0); }
        }
        #pragma omp parallel for
        for(int i=0;i<N;++i) dndt[i]=(n[i]-nold[i])/dt;
        // ---- c: migration (upwind) + diffusion (implicit Dirichlet) ----
        #pragma omp parallel for
        for(int i=1;i<nx-1;++i) for(int j=0;j<ny;++j){
            int jm=(j-1+ny)%ny, jp=(j+1)%ny, id=i*ny+j;
            double vmax=dx/dt;  // CFL cap
            double vxp=-Mmig*(phi[(i+1)*ny+j]-phi[id])/dx;
            double vxm=-Mmig*(phi[id]-phi[(i-1)*ny+j])/dx;
            if(vxp> vmax)vxp= vmax; if(vxp<-vmax)vxp=-vmax;
            if(vxm> vmax)vxm= vmax; if(vxm<-vmax)vxm=-vmax;
            double fx_p=(vxp>0.0?c[id]:c[(i+1)*ny+j])*vxp;
            double fx_m=(vxm>0.0?c[(i-1)*ny+j]:c[id])*vxm;
            double vyp=-Mmig*(phi[i*ny+jp]-phi[id])/dy;
            double vym=-Mmig*(phi[id]-phi[i*ny+jm])/dy;
            if(vyp> vmax)vyp= vmax; if(vyp<-vmax)vyp=-vmax;
            if(vym> vmax)vym= vmax; if(vym<-vmax)vym=-vmax;
            double fy_p=(vyp>0.0?c[id]:c[i*ny+jp])*vyp;
            double fy_m=(vym>0.0?c[i*ny+jm]:c[id])*vym;
            cbuf[id]=c[id]-dt*((fx_p-fx_m)/dx+(fy_p-fy_m)/dy);
            if(cbuf[id]<0.0)cbuf[id]=0.0;
        }
        double xi=sqrt(kappa/(2.0*W));  // interface half-width
        #pragma omp parallel for
        for(int i=1;i<nx-1;++i) for(int j=0;j<ny;++j){
            int jm=(j-1+ny)%ny, jp=(j+1)%ny, id=i*ny+j;
            // deposition velocity V ~ dN/dt; flux F = dN * n_hat (dt cancels), localized at interface
            double gxf=(n[(i+1)*ny+j]-n[id])/dx;
            double gyf=0.25*(n[i*ny+jp]-n[i*ny+jm]+n[(i+1)*ny+jp]-n[(i+1)*ny+jm])/dy;
            double gmf=sqrt(gxf*gxf+gyf*gyf)+1e-9;
            double dnf=0.5*(dN[id]+dN[(i+1)*ny+j]);
            double Fxp=dnf*gxf/gmf;
            double gxm=(n[id]-n[(i-1)*ny+j])/dx;
            double gym=0.25*(n[i*ny+jp]-n[i*ny+jm]+n[(i-1)*ny+jp]-n[(i-1)*ny+jm])/dy;
            double gmm=sqrt(gxm*gxm+gym*gym)+1e-9;
            double dnm=0.5*(dN[id]+dN[(i-1)*ny+j]);
            double Fxm=dnm*gxm/gmm;
            double gyf2=(n[i*ny+jp]-n[id])/dy;
            double gxf2=0.25*(n[(i+1)*ny+j]-n[(i-1)*ny+j]+n[(i+1)*ny+jp]-n[(i-1)*ny+jp])/dx;
            double gm2=sqrt(gyf2*gyf2+gxf2*gxf2)+1e-9;
            double dnf2=0.5*(dN[id]+dN[i*ny+jp]);
            double Fyp=dnf2*gyf2/gm2;
            double gym2=(n[id]-n[i*ny+jm])/dy;
            double gxm2=0.25*(n[(i+1)*ny+j]-n[(i-1)*ny+j]+n[(i+1)*ny+jm]-n[(i-1)*ny+jm])/dx;
            double gm3=sqrt(gym2*gym2+gxm2*gxm2)+1e-9;
            double dnm2=0.5*(dN[id]+dN[i*ny+jm]);
            double Fym=dnm2*gym2/gm3;
            double divF=(Fxp-Fxm)/dx+(Fyp-Fym)/dy;
            cbuf[id]+=xi*0.0*divF;   // antitrapping disabled (needs Karma thin-interface)
        }
        for(int j=0;j<ny;++j){ cbuf[j]=c[j]; cbuf[(nx-1)*ny+j]=c[(nx-1)*ny+j]; }
        helm.solve(c,cbuf,1.0,dt*D,0.0,1.0,(bool)finitemode);
        #pragma omp parallel for
        for(int i=0;i<N;++i) if(c[i]<0)c[i]=0;
        for(int j=0;j<ny;++j){ c[j]=0.0; if(finitemode) c[(nx-1)*ny+j]=c[(nx-2)*ny+j]; else c[(nx-1)*ny+j]=1.0; }

        if(step%200==0){
            vector<double> xs(ny);
            double s=0;
            for(int j=0;j<ny;++j){ double xj=0; for(int i=0;i<nx;++i) if(n[i*ny+j]<0.5){ xj=i*dx; break; } xs[j]=xj; s+=xj; }
            s/=ny;
            double var=0; for(int j=0;j<ny;++j) var+=(xs[j]-s)*(xs[j]-s); var/=ny;
            int nt=0; for(int j=1;j<ny-1;++j) if(xs[j]>xs[j-1]&&xs[j]>xs[j+1]) nt++;
            double tot=0,totc=0; for(int i=0;i<N;++i){ tot+=n[i]; totc+=c[i]; }
            printf("t=%.1fs ifc=%.2f rough=%.3f tips=%d ntot=%.0f totLi=%.1f\n",
                   step*dt,s,sqrt(var),nt,tot,tot+totc/Cm); fflush(stdout);
            {double sdN=0,mdN=0;int mi=0,mj=0;for(int i=0;i<N;++i){sdN+=dN[i];if(dN[i]>mdN){mdN=dN[i];mi=i/ny;mj=i%ny;}}
             printf("   [diag] sum(dN)=%.3e  max(dN)=%.3e at x=%.1fum n=%.2f c=%.2f\n",sdN,mdN,mi*dx,n[mi*ny+mj],c[mi*ny+mj]);}
        }
        if(step%500==0){
            char fn[64]; snprintf(fn,64,"ddft_two_n_%03d.dat",snap);
            FILE*fp=fopen(fn,"wb"); fwrite(n.data(),sizeof(double),N,fp); fclose(fp);
            snprintf(fn,64,"ddft_two_c_%03d.dat",snap);
            fp=fopen(fn,"wb"); fwrite(c.data(),sizeof(double),N,fp); fclose(fp);
            snprintf(fn,64,"ddft_two_phi_%03d.dat",snap);
            fp=fopen(fn,"wb"); fwrite(phi.data(),sizeof(double),N,fp); fclose(fp);
            snap++;
        }
    }
    fprintf(stderr,"DONE %.1f s\n", chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count());
    fftw_destroy_plan(pA); fftw_destroy_plan(pB); fftw_destroy_plan(pC); fftw_destroy_plan(inv);
    return 0;
}
