# MEPID-Join [VLDB '27]

This repository contains the source code for MEPID-Join [VLDB '27], a memory-efficient in-memory join algorithm for capacity-limited Processing-in-DIMM architectures (e.g., UPMEM DIMM).
If you find MEPID-Join useful to your research, please cite:

```bibtex
@article{lee2027mepidjoin,
  author  = {Suhyun Lee and Hanna Cha and Chaemin Lim and Hanjun Kim and Youngsok Kim},
  title   = {{MEPID-Join: Scaling Processing-in-DIMM Joins Using Adaptive Key-Level Replication}},
  journal = {Proceedings of the VLDB Endowment (PVLDB)},
  volume  = {20},
  number  = {2},
  year    = {2026}
}
```

## System Configuration

- Intel Xeon Gold 5215 CPU
- 1 DDR4 channel with 2 64-GB DDR4-2400 DIMMs
- 4 DDR4 channels, each with two UPMEM DIMMs
- Ubuntu 18.04 (x64)

## Prerequisites

- g++
- python3.6
- matplotlib
- numpy
- Pandas
- Scipy
- The driver for UPMEM SDK (version 2023.2, available from the [UPMEM website](https://sdk.upmem.com/).)

## Directories

- src/ # The source codes for libmepidjoin.so
- test/ # The test srcs for MEPID-Join
- upmem-2023.2.0-Linux-x86_64/ # Contains the modified upmem sdk version 2023.2 for mepid-join

## Download Upmem SDK
```
# firstly download upmem sdk then,
cp -r {your upmem sdk dir}/lib {your mepid-join dir}/upmem-2023.2.0-Linux-x86_64/;
cp -r {your upmem sdk dir}/share {your mepid-join dir}/upmem-2023.2.0-Linux-x86_64/;
```

## Environment Setup
```
cd {your mepid-join dir};
source ./scripts/upmem_env.sh
```

## Build UPMEM SDK
```
cd {your mepid-join dir};
cd upmem-2023.2.0-Linux-x86_64/src/backends/;
./load.sh
```

## Build MEPID-Join Library
```
cd {your mepid-join dir};
mkdir lib; make lib -j
```

## Build tests for MEPID-Join
```
cd {your mepid-join dir};
make test -j
```

## Run tests for MEPID-Join

### Required Arguments
- -s
    - Input size of table R [tuples]
- -r
    - The number of ranks to use
- -z
    - Zipf factor to use; should pass the argument as Zipf factor * 100 (e.g., if you want to test zipf factor 1.0; -z 100)
- -t
    - R-S Ratio to test; (e.g., if you want to test R:S=1:2; -t 2)
- -R
    - The number of rank set to use
- -B
    - The number of bank set to use
- -e
    - PID-accelerated join algorithm to use; -e 0 selects PID-Join when -B 1 or SPID-Join when -B > 1, while -e 1 selects MEPID-Join (e.g., -e 1)

<!-- ### For example,
```
cd {your mepid-join dir};
./mepidjoin_test.bin -s 500000 -r 16 -z 0 -t 8 -R 1 -B 1
```

## Sample Script for running MEPID-Join

To check if MEPID-Join works well for the given example used in the paper, (|S|=4Mtuples, |R|:|S|=1:1)
```
cd {your mepid-join dir};
bash ./scripts/run_script.sh
cd results;
``` -->