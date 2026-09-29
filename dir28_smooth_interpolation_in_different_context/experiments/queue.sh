#!/bin/bash
cd "$(dirname "$0")"
for c in s1 s2 s3 s4; do MPLBACKEND=Agg python sweep.py $c > ../results/sweep_$c.log 2>&1; done
echo done > ../results/queue.done
