from data import make_ds
from models import build_baseline, compile_model
from train_utils import set_seed, list_trainable, fit_stage

set_seed()
train_ds, val_ds = make_ds("train", train=True), make_ds("val")
model = compile_model(build_baseline(), lr=1e-3)
list_trainable(model, "baseline")
fit_stage(model, "baseline", train_ds, val_ds, epochs=40)