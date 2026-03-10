from src.pruning.random_pruning import prune as random_prune
from src.pruning.magnitude import prune as magnitude_prune
from src.pruning.wanda import prune as wanda_prune
from src.pruning.utils import verify_sparsity, save_pruned_model, get_calibration_data
