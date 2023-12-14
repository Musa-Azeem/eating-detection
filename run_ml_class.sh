#!/bin/bash

# Train classifier

epochs=100

run_func="train_resnet"
outdir="dev/resnet"

args="epochs=${epochs}, outdir='${outdir}', device='cuda:1', label='resnet'"

env/bin/python3 -c "from lib.run import $run_func; $run_func($args)" &
wait

echo "Done."
