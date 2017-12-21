[CS515-2017 Fall] project 1
- Wangchao Sheng, NetID: ws333, RUID: 169006238

To compile:
    $ make clean && make

To run:
    $ ./spmv -mat matrix.mtx -ivec vector.mtx -alg segment/ftp/graph -blknum 8 -blksize 256

There are three options for -alg, segment - segment scan, without improvement
                                  ftp - segment scan, with first touch packing algorithm
                                  graph - segment scan, with graph based packing algorithm
