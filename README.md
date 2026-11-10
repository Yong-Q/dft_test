# dft_test — Li whisker/dendrite DDFT (selection)

## whisker_v3/  (MOST CORRECT / primary)
Conserved-interface whisker: SEI pinhole passivation (flat frozen), field-driven deposition on the liquid side
(electrolyte-internal normal current), Li+ transport with Sand (concentration polarization), physical rate (~nm/s).
- constant-width pinhole whisker, advances without flattening; global φ and c (no masking/mirror artifacts).
- height-function representation (sharp interface). Isotropic (no crystallographic anisotropy yet).
- whisker_v3_out/: same physics + c-field dump (for figures/analysis).

## ddft_two.cpp  (the "two")
Cahn-Hilliard (Model B, bi-Laplacian smooth, Eyre) + SEI channel + c coupling. 
- HAS 4-fold crystallographic anisotropy (-eps eps4, kappa(theta)=kappa(1+eps4 cos4theta)) -> faceting/branching tendency.
- KNOWN ISSUES: (a) c-migration pile-up artifact (set Mmig=0 to avoid; then it flattens because the small-bump field is weak);
  (b) reaction uses solid-cell phi (equipotential metal -> eta const), so it does NOT concentrate at the tip;
  (c) unphysical tuning was used in production (Rfac=150) to force growth.
- Kept because it has the anisotropy machinery for future faceting/dendritic work.

Build: g++ -O3 -fopenmp -march=native ddft_two.cpp -o ddft_two -I$FFTW/include -L$FFTW/lib -lfftw3_threads -lfftw3 -lm
