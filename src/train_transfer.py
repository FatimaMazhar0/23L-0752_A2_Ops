from data import make_ds
from models import build_transfer, unfreeze_last_block, compile_model
from train_utils import set_seed, list_trainable, fit_stage

set_seed()
train_ds, val_ds = make_ds("train", train=True), make_ds("val")

# Stage 1: frozen backbone, new head, normal learning rate
model = compile_model(build_transfer(), lr=1e-3)
list_trainable(model, "mnv2_stage1")
fit_stage(model, "mnv2_stage1", train_ds, val_ds, epochs=25)   # best weights restored

# Stage 2: unfreeze last block only, much smaller learning rate
unfreeze_last_block(model)
model = compile_model(model, lr=1e-5)                           # recompile is required
list_trainable(model, "mnv2_stage2")
fit_stage(model, "mnv2_stage2", train_ds, val_ds, epochs=20)