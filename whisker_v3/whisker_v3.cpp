// whisker_v3: sharp-interface (height-function) electrochemical whisker growth.
// interface x_if(y); solid where x<x_if (phi=eta Dirichlet); Laplace in electrolyte.
// v_n = kdep*j_n - kdep_gt*d0*curv   (tiny physical d0); Li+ diffusion + consumption.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
using namespace std;
int main(int argc,char**argv){
 int nx=300,ny=180,nsteps=20000,nphi=200,nc=25,nout=500;
 double dx=0.2, dt=0.01;
 double x0=20.0, ww=0.6, wh0=0.4, eta=-0.45, phifar=0.0;
 double gap_ff=0.0;         // far-field gap (um); if >0 use far-field BC at right (for local fine domain)
 double sig_l=1.0, D=20.0, Cm=76.9;
 double kdep=8.0;            // deposition velocity coeff: v_n = kdep*j_n
 double d0=0.002;            // physical Li capillary length ~2nm (um)  -> negligible GT
 double kgt=1.0; double wp=0.6;  // SEI pinhole width
 for(int a=1;a<argc;++a){
   if(!strcmp(argv[a],"-nx"))nx=atoi(argv[++a]);
   else if(!strcmp(argv[a],"-ny"))ny=atoi(argv[++a]);
   else if(!strcmp(argv[a],"-dx"))dx=atof(argv[++a]);
   else if(!strcmp(argv[a],"-n"))nsteps=atoi(argv[++a]);
   else if(!strcmp(argv[a],"-nphi"))nphi=atoi(argv[++a]);
   else if(!strcmp(argv[a],"-dt"))dt=atof(argv[++a]);
   else if(!strcmp(argv[a],"-x0"))x0=atof(argv[++a]);
   else if(!strcmp(argv[a],"-ww"))ww=atof(argv[++a]);
   else if(!strcmp(argv[a],"-wh0"))wh0=atof(argv[++a]);
   else if(!strcmp(argv[a],"-eta"))eta=atof(argv[++a]);
   else if(!strcmp(argv[a],"-D"))D=atof(argv[++a]);
   else if(!strcmp(argv[a],"-kdep"))kdep=atof(argv[++a]);
   else if(!strcmp(argv[a],"-d0"))d0=atof(argv[++a]);
   else if(!strcmp(argv[a],"-wp"))wp=atof(argv[++a]);
   else if(!strcmp(argv[a],"-gap"))gap_ff=atof(argv[++a]);
   else if(!strcmp(argv[a],"-nout"))nout=atoi(argv[++a]);
 }
 int yc=ny/2,N=nx*ny; double Lx=nx*dx;
 if(gap_ff>0) phifar=eta*(1.0-(Lx-x0)/gap_ff);   // far-field potential at right boundary (local fine domain)
 vector<double> xif(ny), xif_old(ny);
 vector<double> phi(N), c(N,1.0), cn(N);
 auto jp=[&](int j){return (j+1)%ny;}; auto jm=[&](int j){return (j-1+ny)%ny;};
 // init interface: film at x0, whisker column |y|<ww/2 to x0+wh0
 for(int j=0;j<ny;++j){ double y=(j-yc)*dx; xif[j]=x0 + (fabs(y)<0.5*ww? wh0:0.0); }
 // c init: 1 in electrolyte
 fprintf(stderr,"v3 sharp-interface nx=%d ny=%d dx=%.2f eta=%.2f kdep=%.1f d0=%.4fum\n",nx,ny,dx,eta,kdep,d0);
 for(int step=0;step<=nsteps;++step){
   // ---- 1. potential: solid (x<xif)=eta (Dirichlet), electrolyte Laplace, far=phifar ----
   for(int i=0;i<nx;++i)for(int j=0;j<ny;++j){ int id=i*ny+j; id=id;
     if(i*dx < xif[j]) phi[id]=eta; }
   for(int j=0;j<ny;++j) phi[(nx-1)*ny+j]=phifar;
   for(int it=0;it<nphi;++it){
     for(int i=1;i<nx-1;++i)for(int j=0;j<ny;++j){int id=i*ny+j;
       if(i*dx < xif[j]) {phi[id]=eta; continue;}
       double val=0.25*(phi[(i+1)*ny+j]+phi[(i-1)*ny+j]+phi[i*ny+jp(j)]+phi[i*ny+jm(j)]);
       phi[id]+=1.85*(val-phi[id]);
     }
     for(int j=0;j<ny;++j) phi[(nx-1)*ny+j]=phifar;
   }
   // ---- 2. normal current at interface per column; advance xif (deposition - tiny GT) ----
   for(int j=0;j<ny;++j){
     if(fabs((j-yc)*dx) > 0.5*wp){ xif[j]=x0; continue; }   // SEI-passivated flat: frozen, no deposition
     int ie=(int)(xif[j]/dx)+1; if(ie<2)ie=2; if(ie>nx-2)ie=nx-2;          // first electrolyte node
     double jn=sig_l*((phi[(ie+1)*ny+j]-phi[ie*ny+j])/dx);                 // STABLE field: electrolyte-internal gradient
     if(jn<0)jn=0;
     double vn=kdep*jn;                                                    // field-driven deposition (fixed current, like sand_2d)
     double need=vn*dt*Cm;                                                 // Li+ consumed by depositing vn (Cm = solid/solution conversion)
     if(c[ie*ny+j]>=need){ c[ie*ny+j]-=need; }                             // enough Li: deposit, consume
     else { vn *= c[ie*ny+j]/(need+1e-30); c[ie*ny+j]=0.0; }               // Sand: Li runs out -> cap growth, deplete c to ~0
     xif[j]+= vn*dt;
     if(xif[j]<x0) xif[j]=x0;                                             // don't dissolve the film
     if(xif[j]>Lx-2*dx) xif[j]=Lx-2*dx;
   }
   // ---- 2b. surface diffusion (Mullins) on the pinhole interface, Neumann (no-flux) at SEI wall -> rounds tip, no collapse ----
   { vector<double> xs=xif; double B=0.10;
     for(int j=0;j<ny;++j){ if(fabs((j-yc)*dx)>0.5*wp) continue;
       double xl = (fabs((jm(j)-yc)*dx)>0.5*wp)? xs[j]: xs[jm(j)];   // mirror at frozen wall (Neumann)
       double xr = (fabs((jp(j)-yc)*dx)>0.5*wp)? xs[j]: xs[jp(j)];
       xif[j]+= B*(xl+xr-2.0*xs[j])*0.5;
     } }
   // ---- 3. Li+ explicit diffusion (sand_2d style, sub-cycled for CFL) ----
   { double dtc=0.24*dx*dx/(4.0*D); if(dtc<=0)dtc=1e9; int nsub=(int)(dt/dtc)+1; double h=dt/nsub;
     for(int sub=0;sub<nsub;++sub){
       for(int i=0;i<N;++i) cn[i]=c[i];
       for(int i=1;i<nx-1;++i)for(int j=0;j<ny;++j){int id=i*ny+j;
         if(i*dx < xif[j]){ cn[id]=0.0; continue; }                     // solid: no Li
         double lap=0; int nb[4]={(i+1)*ny+j,(i-1)*ny+j,i*ny+jp(j),i*ny+jm(j)};
         for(int q=0;q<4;++q){int k=nb[q]; if(!((k/ny)*dx < xif[k%ny])) lap += (c[k]-c[id]);}  // Neumann at solid
         cn[id]=c[id]+ h*D/(dx*dx)*lap;
       }
       for(int j=0;j<ny;++j) cn[(nx-1)*ny+j]=1.0;                       // reservoir
       for(int i=0;i<N;++i){ c[i]=cn[i]; if(c[i]<0)c[i]=0; }
     }
   }
   // ---- output ----
   if(step%200==0){
     double hmax=*max_element(xif.begin(),xif.end());
     int jc2=yc;
     {int ic=(int)(xif[yc]/dx)+1; double jc_n=sig_l*((phi[(ic+1)*ny+yc]-phi[ic*ny+yc])/dx); int ie=(int)(xif[yc]/dx)+1; double cav=std::min(1.0,std::max(0.0,c[ie*ny+yc])); fprintf(stderr,"t=%.1fs tip_x=%.2f xifE_flat=%.2f jn_tip=%.4f c_tip=%.3f\n",step*dt,xif[yc],xif[10],jc_n,cav);}
   }
   if(step%nout==0){ int k=step/nout; char fn[64];
     snprintf(fn,64,"v3_xif_%03d.dat",k); FILE*f=fopen(fn,"wb");fwrite(xif.data(),8,ny,f);fclose(f);
     snprintf(fn,64,"v3_phi_%03d.dat",k); f=fopen(fn,"wb");fwrite(phi.data(),8,N,f);fclose(f);
   }
 }
 fprintf(stderr,"DONE\n"); return 0;
}
