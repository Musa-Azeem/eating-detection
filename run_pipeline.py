import pandas as pd
import time
import datetime
import plotly.express as px
from pathlib import Path
import torch
from lib.models import MLP2hl
from lib.modules import evaluate_loop, plot_and_save_cm, summary, window_session, pad_for_windowing, predict_and_plot_pretty_session
from lib.utils import get_bouts_smoothed
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WINSIZE = 101
DEVICE = 'cuda:1'

data_dir = Path('../data/andrew/new')
acceleration = pd.read_csv(data_dir / 'acceleration.csv',skiprows=1).rename({'x':'x_acc', 'y':'y_acc', 'z':'z_acc'}, axis=1)
acceleration_start_time_seconds = float(pd.read_csv(data_dir / 'acceleration.csv',nrows=1,header=None).iloc[0,0].split()[-1])/1000
acceleration.timestamp = ((acceleration.timestamp - acceleration.timestamp[0])*1e-9)+acceleration_start_time_seconds

start = int(datetime.datetime(2023, 10, 26, 16, 20, 0).strftime('%s'))
end = int(datetime.datetime(2023, 10, 26, 16, 37, 0).strftime('%s'))
acceleration['label'] = 0
acceleration.loc[(acceleration.timestamp > start) & (acceleration.timestamp < end),'label'] = 1

Xte = torch.Tensor(acceleration[['x_acc','y_acc','z_acc']].values)
yte = torch.Tensor(acceleration['label'].values).unsqueeze(1)

Xte = pad_for_windowing(Xte, WINSIZE)
Xte = window_session(Xte, WINSIZE)

model = MLP2hl([20,20], WINSIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model.load_state_dict(torch.load(Path('dev/mlp2hl/best_model.pt')))

testloader = DataLoader(TensorDataset(Xte,yte), batch_size=64)
ys,metrics = evaluate_loop(model, criterion, testloader, DEVICE)
plot_and_save_cm(ys['true'], ys['pred'])
summary(metrics)

torch.save(ys['pred'], 'pred.pt')
pd.DataFrame({'pred': ys['pred'].flatten().numpy()}).to_csv('pred.csv', index=False)
