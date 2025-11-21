# data taken from "opesexp_leo_T2d_NVT_bkp"

outdir=res_fromKERNELS_reference
mkdir -p $outdir

../calcFES.py \
  -o $outdir/dZ.dat \
  --colvar COLVAR_merged.dat --temp 300 \
  --bin 100 --sigma 0.05 --cv dT.z \
  --bias opes.bias --plot

../calcFES.py \
  -o $outdir/dZ_16block.dat \
  --colvar COLVAR_merged.dat --temp 300 \
  --bin 100 --sigma 0.05 --cv dT.z \
  --bias opes.bias --blocks 16 --plot

../calcFES.py \
  -o $outdir/dZ_500000stride.dat \
  --colvar COLVAR_merged.dat --temp 300 \
  --bin 100 --sigma 0.05 --cv dT.z \
  --bias opes.bias --stride 500000 --plot


../calcFES.py \
  -o $outdir/2D_50000stride.dat \
  --colvar COLVAR_merged.dat --temp 300 \
  --bin 50,50 --sigma 0.05,5 --cv dT.z,tiltAvg \
  --bias opes.bias --stride 500000 --plot

../calcFES.py \
  -o $outdir/2D_16block.dat \
  --colvar COLVAR_merged.dat --temp 300 \
  --bin 50,50 --sigma 0.05,5 --cv dT.z,tiltAvg \
  --bias opes.bias --blocks 16 --plot

../calcFES.py \
  -o $outdir/2D.dat \
  --colvar COLVAR_merged.dat --temp 300 \
  --bin 50,50 --sigma 0.05,5 --cv dT.z,tiltAvg \
  --bias opes.bias --plot
