from pathlib import Path
from lib.models import RegNetMAE, CosineEmbeddingLossPositive
from lib.data.dataloading import load_raw
from lib.modules import optimization_loop_xonly
from lib.config import RAW_DIR
import torch
from torch import nn

def train_mae_9(CONFIG):
    p_dropout = 0.1
    model = RegNetMAE(
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
        maskpct=CONFIG['MASKPCT']
    ).to(CONFIG['DEVICE'])

    criterion = nn.MSELoss()
    # criterion = CosineEmbeddingLossPositive()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    trainloader, testloader = load_raw(
        RAW_DIR,
        CONFIG['WINDOW_SIZE'],
        # n_hours=20,
        test_size=CONFIG['TEST_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle_test=True,
        chunk_len_hrs=0.25,
        stride=CONFIG['WINDOW_STRIDE']
    )

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
    )
    optimization_loop_xonly(
        model,
        trainloader,
        testloader,
        criterion,
        optimizer,
        epochs=2500,
        patience=500,
        config=CONFIG,
        continue_training=False,
        device=CONFIG['DEVICE'],
        outdir=f'dev/9_regnet-mae/dev/{outdir}',
        writer=f'runs/9_regnet-mae/{outdir}'
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
if __name__ == '__main__':
    for mask_pcti in [0.0, 0.25, 0.5, 0.75]:
        for dmodeli in [32, 512]:
            CONFIG['MASKPCT'] = mask_pcti
            CONFIG['DMODEL'] = dmodeli
            try:
                train_mae_9(CONFIG)
            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise e
                
                print(e)
                for bi in [64,32]:
                    CONFIG['BATCH_SIZE'] = bi
                    try:
                        train_mae_9(CONFIG)
                        break
                    except RuntimeError as e:
                        if not 'CUDA out of memory' in str(e):
                            raise e
                        print(e)