import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# https://pytorch.org/docs/stable/optim.html
# loss_func = F.cross_entropy

# https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751/2
# https://neptune.ai/blog/pytorch-loss-functions
mse_loss_func = nn.MSELoss()
# mae_loss_func = nn.L1Loss()
# cross_entropy_loss_func = nn.CrossEntropyLoss()
# hinge_embedding_loss_func = nn.HingeEmbeddingLoss() # binary classification
# triplet_margin_loss_func = nn.TripletMarginLoss()
# kl_loss_func = nn.KLDivLoss(reduction = 'batchmean')
# n_l_l_loss_func = torch.nn.NLLLoss # Needs softmax function as an output activation layer

loss_functions = [
  {"name": "mse_loss_func", "function": mse_loss_func},
  # {"name": "mae_loss_func", "function": mae_loss_func},
  # {"name": "cross_entropy_loss_func", "function": cross_entropy_loss_func}, # TODO: IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
  # {"name": "hinge_embedding_loss_func", "function": hinge_embedding_loss_func},
  # {"name": "triplet_margin_loss_func", "function": triplet_margin_loss_func}, # TODO: output = triplet_loss(anchor, positive, negative)
  # {"name": "kl_loss_func", "function": kl_loss_func},
  # {"name": "n_l_l_loss_func", "function": n_l_l_loss_func} # TODO: RuntimeError: bool value of Tensor with more than one value is ambiguous
]

def fetch_loss_opt_func(config, model):
  betas = (config["beta_1"], config["beta_2"])

  # Optimiser
  # See: https://discuss.pytorch.org/t/function-for-picking-an-optimizer/106798
  # optimizers = {
  #   'sgd': optim.SGD(*args, **kwargs),
  #   'asgd': optim.ASGD(*args, **kwargs),
  #   'adam': optim.Adam(*args, **kwargs),
  #   Adadelta
  #   Adagrad
  #   SparseAdam
  #   LBFGS
  #   'adamw': optim.AdamW(*args, **kwargs),
  #   'adamax': optim.Adamax(*args, **kwargs),
  #   'rms': optim.RMSprop(*args, **kwargs),
  #   RMSprop
  #   'rprop': optim.Rprop(*args, **kwargs),
  #   Rprop
  #   'adah': AdaHessian(*args, **kwargs)
  # }

  if (config["opt_name"] == "SGD"):
    opt = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
  elif (config["opt_name"] == "Adam"):
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=betas, eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
  elif (config["opt_name"] == "asgd"):
    # torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    opt = torch.optim.asgd(model.parameters(), lr=config["lr"], betas=betas, eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
  elif (config["opt_name"] == "Adadelta"):
    # torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    opt = torch.optim.Adadelta(model.parameters(), lr=config["lr"], betas=betas, eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
  # torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
  elif (config["opt_name"] == "adamw"):
    opt = torch.optim.adamw(model.parameters(), lr=config["lr"], betas=betas, eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    # torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
  elif (config["opt_name"] == "adamax"):
    opt = torch.optim.adamax(model.parameters(), lr=config["lr"], betas=betas, eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    # torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  elif (config["opt_name"] == "rms"):
    opt = torch.optim.rms(model.parameters(), lr=config["lr"], betas=betas, eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    # torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    # torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
  elif (config["opt_name"] == "rprop"):
    opt = torch.optim.rprop(model.parameters(), lr=config["lr"], betas=betas, eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    # torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
  elif (config["opt_name"] == "adah"):
    opt = torch.optim.adah(model.parameters(), lr=config["lr"], betas=betas, eps=config["epsilon"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])

  loss_func = loss_functions["name" == config["loss_name"]]["function"]

  return [opt, loss_func]
