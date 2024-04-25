from process.produce_graph import creat_dataloader
from model.model import TCAN
from process.produce_graph import obtain_N
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from process.setting import OPT
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = OPT()
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=True,
)

train_loader = creat_dataloader(0, 32,  True)
val_loader = creat_dataloader(1, 32, False)
test_loader = creat_dataloader(2, 32, False)
N = obtain_N() + 1

model = TCAN(N, opt.input_dim, opt.out_dim).to(device)
print(model)
trainer = pl.Trainer(max_epochs=10, callbacks=[early_stop_callback], gradient_clip_val=opt.max_grad_clip)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader, verbose=True)

#display

