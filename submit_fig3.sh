#!/bin/bash
#SBATCH -J fig3run
#SBATCH -p amd_m9_768
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 256
#SBATCH -t 04:00:00
#SBATCH -o fig3run_%j.out
cd /publicfs10/fs10-m9/home/m9s005258/qy/DDFT/code/cpp/run_fig3new
export LD_LIBRARY_PATH=/publicfs10/fs10-share/soft/share-soft/fftw/fftw-3.3.10-openmpi410-gcc14.2.0/install/lib:$LD_LIBRARY_PATH
rm -f ddft_two_*.dat
export OMP_NUM_THREADS=256
../ddft_two -nx 300 -ny 180 -n 80000 -modelA 0 -Mn 0.1 -D 300 -Rfac 150 -whisker 0.6 0.4 -ccdep -snap 3000
echo DONE_FIG3
