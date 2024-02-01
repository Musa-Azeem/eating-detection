from pathlib import Path
from lib.models import RegNetMAE, CosineEmbeddingLossPositive
from lib.data.dataloading import load_raw
from lib.modules import optimization_loop_xonly
from lib.config import RAW_DIR
import torch
from torch import nn
import json

def train_mae_9_class(CONFIG, weights_file, freeze):
    p_dropout = 0.1
    model = RegNetClassifier(
        winsize=CONFIG['WINDOW_SIZE'], 
        in_channels=3, 
        stem_out_c=4, 
        d=CONFIG['DEPTHI'], 
        w=CONFIG['WIDTHI'], 
        d_model=CONFIG['DMODEL'], 
        b=1, 
        g=2, 
        p_dropout=p_dropout, 
        ntrans=CONFIG['NTL'], 
        nhead=2,
        maskpct=CONFIG['MASKPCT'],
        weights_file=weights_file,
        freeze=freeze
    ).to('cuda:1')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    nursing_trainloader, nursing_testloader = load_nursing_5_class(
        range(11,71), 
        CONFIG['WINDOW_SIZE'], 
        test_size=CONFIG['TEST_SIZE'], 
        batch_size=CONFIG['BATCH_SIZE'],
        stride=CONFIG['WINDOW_STRIDE'],
    )

    pretrained = 'pretrained' if weights_file else 'random'
    frozen = 'frozen' if freeze else 'unfrozen'
    outdir = (
        f'{str(type(model)).split(".")[-1][:-2]}'
        f'_lossfn-{str(type(criterion)).split(".")[-1][:-2]}'
        f"_win{CONFIG['WINDOW_SIZE']}"
        f"_stride{CONFIG['WINDOW_STRIDE']}"
        f"_so{model.stem_out_c}_d{model.d_str}_w{model.w_str}"
        f"_b{model.b}_g{model.g}"
        f'_p{p_dropout}'
        f'_ntl{model.ntrans}_nth{model.nhead}_dmodel{model.d_model}'
        f'_maskpct{model.maskpct}'
        f'_{pretrained}_{frozen}'
    )
    optimization_loop_multi_class(
        model,
        trainloader,
        testloader,
        criterion,
        optimizer,
        epochs=2500,
        patience=500,
        device=CONFIG['DEVICE'],
        outdir=f'dev/9_regnet-mae/classifiers/{outdir}',
        writer=f'runs/9_regnet-mae-classifiers/{outdir}'
    )

CONFIG = {
    'WINDOW_SIZE':3901,
    'WINDOW_STRIDE':3901,
    'BATCH_SIZE':128,
    'LEARNING_RATE':3e-4,
    'TEST_SIZE':0.2,
    'DEVICE':'cuda:1',
    'DEPTHI': [1],
    'WIDTHI': [64],
    'NTL': 2,
    'DMODEL': 256,
    'MASKPCT': 0.25
}

def try_wrapper(train_func, CONFIG, weights_file, freeze):
    try:
        train_func(CONFIG, weights_file, freeze)
    except RuntimeError as e:
        if not 'CUDA out of memory' in str(e):
            raise e
        print(e)
        CONFIG = CONFIG.copy()
        for bi in [64,32]:
            CONFIG['BATCH_SIZE'] = bi
            try:
                train_func(CONFIG, weights_file, freeze)
                break
            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise e
                print(e)

if __name__ == '__main__':
    for autoencoder_dir in [
        Path('/home/musa/eating/eating-detection/dev/9_regnet-mae/dev/RegNetMAE_lossfn-MSELoss_win3901_stride3901_so4_d1_w64_b1_g2_p0.1_ntl2_nth2_dmodel64_maskpct0.0'),
        Path('/home/musa/eating/eating-detection/dev/9_regnet-mae/dev/RegNetMAE_lossfn-MSELoss_win3901_stride3901_so4_d1_w64_b1_g2_p0.1_ntl2_nth2_dmodel64_maskpct0.25'),
        Path('/home/musa/eating/eating-detection/dev/9_regnet-mae/dev/RegNetMAE_lossfn-MSELoss_win3901_stride3901_so4_d1_w64_b1_g2_p0.1_ntl2_nth2_dmodel128_maskpct0.5'),
        Path('/home/musa/eating/eating-detection/dev/9_regnet-mae/dev/RegNetMAE_lossfn-MSELoss_win3901_stride3901_so4_d1_w64_b1_g2_p0.1_ntl2_nth2_dmodel64_maskpct0.75')
    ]:
        CONFIG = json.load((autoendoer_dir / 'config.json').open())
        weights_file = autoencoder_dir / 'best_model.pt'

        try_wrapper(CONFIG, weights_file, True)
        try_wrapper(CONFIG, weights_file, False)
        try_wrapper(CONFIG, None, False)
