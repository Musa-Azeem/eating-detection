#!/bin/bash

# Train autoencoder

# ae_args="epochs=100, outdir='dev/8_mae/delta_mask25', device='cuda:0', maskpct=0.25, label='Masked Autoencoder'"
run_func="train_mae_8"
# env/bin/python3 -c "from lib.run import $run_func; $run_func($ae_args)"
# wait

# ae_args="epochs=100, outdir='dev/8_mae/delta_mask50', device='cuda:0', maskpct=0.50, label='Masked Autoencoder'"
# env/bin/python3 -c "from lib.run import $run_func; $run_func($ae_args)"
# wait

ae_args="epochs=100, outdir='dev/8_mae/delta_mask75_10hrs', device='cuda:0', maskpct=0.75, label='Masked Autoencoder'"
env/bin/python3 -c "from lib.run import $run_func; $run_func($ae_args)"

# # Train classifier in 4 ways

# # 1. Train classifier with pretrained, frozen encoder
# class1_args="epochs=100, outdir='dev2/5_encoderclass-resnet_1-pretrained-frozen', device='cuda:0', autoencoder_dir='dev2/5_autoencoder-resnet', freeze=True, label='1. Pretrained, frozen encoder'"

# # 2. Train classifier with pretrained, unfrozen encoder
# class2_args="epochs=100, outdir='dev2/5_encoderclass-resnet_2-pretrained-unfrozen', device='cuda:1', autoencoder_dir='dev2/5_autoencoder-resnet', freeze=False, label='2. Pretrained, unfrozen encoder'"

# # 3. Train classifier with untrained, frozen encoder
# class3_args="epochs=100, outdir='dev2/5_encoderclass-resnet_3-untrained-frozen', device='cuda:0', autoencoder_dir=None, freeze=True, label='3. Untrained, frozen encoder'"

# # 4. Train classifier with untrained, unfrozen encoder
# class4_args="epochs=100, outdir='dev2/5_encoderclass-resnet_4-untrained-unfrozen', device='cuda:1', autoencoder_dir=None, freeze=False, label='4. Untrained, unfrozen encoder'"

# # Train first two
# python3 -c "from lib.run import train_encoderclassifier; train_encoderclassifier($class1_args)" &
# python3 -c "from lib.run import train_encoderclassifier; train_encoderclassifier($class2_args)" &
# wait

# # Train last two
# python3 -c "from lib.run import train_encoderclassifier; train_encoderclassifier($class3_args)" &
# python3 -c "from lib.run import train_encoderclassifier; train_encoderclassifier($class4_args)" &
# wait

echo "Done."