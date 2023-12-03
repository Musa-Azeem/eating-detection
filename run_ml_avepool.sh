#!/bin/bash

# Train autoencoder

autoencoder_outdir="dev/5_resnetautoencoder/5_autoencoder-resnet"


# Train classifier in 4 ways

epochs=100

run_func="train_encoderclassifier_avgpool"
class_outdir="dev/5_resnetautoencoder/encoder_class-avg"

# 1. Train classifier with pretrained, frozen encoder
class1_args="epochs=${epochs}, outdir='${class_outdir}/1-pretrained-frozen', device='cuda:0', autoencoder_dir='${autoencoder_outdir}', freeze=True, label='1. Pretrained, frozen encoder'"

# 2. Train classifier with pretrained, unfrozen encoder
class2_args="epochs=${epochs}, outdir='${class_outdir}/2-pretrained-unfrozen', device='cuda:1', autoencoder_dir='${autoencoder_outdir}', freeze=False, label='2. Pretrained, unfrozen encoder'"

# 3. Train classifier with untrained, frozen encoder
class3_args="epochs=${epochs}, outdir='${class_outdir}/3-untrained-frozen', device='cuda:0', autoencoder_dir=None, freeze=True, label='3. Untrained, frozen encoder'"

# 4. Train classifier with untrained, unfrozen encoder
class4_args="epochs=${epochs}, outdir='${class_outdir}/4-untrained-unfrozen', device='cuda:1', autoencoder_dir=None, freeze=False, label='4. Untrained, unfrozen encoder'"

# Train first two
env/bin/python3 -c "from lib.run import $run_func; $run_func($class1_args)" &
env/bin/python3 -c "from lib.run import $run_func; $run_func($class2_args)" &
wait

# Train last two
env/bin/python3 -c "from lib.run import $run_func; $run_func($class3_args)" &
env/bin/python3 -c "from lib.run import $run_func; $run_func($class4_args)" &
wait

echo "Done."
