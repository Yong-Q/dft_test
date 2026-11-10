#!/bin/bash
# EXACT reproduction of FIG3_whisker_naturenano.png data (recovered from session transcript)
# conserved Model-B DDFT + 0.6um SEI-pinhole whisker + self-consistent sigma(c) potential
set -e
export LD_LIBRARY_PATH=/publicfs10/fs10-share/soft/share-soft/fftw/fftw-3.3.10-openmpi410-gcc14.2.0/install/lib:$LD_LIBRARY_PATH
OUT=${1:-run_whisker_cc}
cd "$(dirname "$0")"
[ -x ./ddft_two ] || g++ -O3 -fopenmp -march=native ddft_two.cpp -o ddft_two \
   -I$FFTW/include -L$FFTW/lib -lfftw3_threads -lfftw3 -lm
mkdir -p "$OUT"; rm -f ddft_two_n_*.dat ddft_two_c_*.dat ddft_two_phi_*.dat
OMP_NUM_THREADS=16 ./ddft_two -nx 300 -ny 180 -n 30000 -modelA 0 -Mn 0.5 -D 20 \
   -whisker 0.6 0.4 -ccdep -snap 3000
mv ddft_two_n_*.dat ddft_two_c_*.dat ddft_two_phi_*.dat "$OUT"/
echo "done -> $OUT   (reshape nx=300, ny=180 for plotting)"
