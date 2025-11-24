# preen
Collection of tools to manipulate plumed files

## CLI highlights
- `preen colvar merge` now interleaves lines by default (`--concat` switches to simple append) and always writes `BASENAME_merged.dat` unless you pass `--output`.
- Header validation is strict by default; use `--allow-header-mismatch` to proceed with differing header lengths (a warning is emitted).
- `preen colvar plot` and `preen colvar reweight` reuse the merged data and validate requested columns before heavy work.
- `preen fes fromstate` computes FES directly from a STATE/KERNELS file (see `preen fes fromstate --help` for options like `--temp`, `--grid-bin`, `--plot`).
