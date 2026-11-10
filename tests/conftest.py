"""Study 01's trainer tests need MLX (Apple silicon); skip that file where MLX is not installed."""
import importlib.util

collect_ignore = [] if importlib.util.find_spec("mlx") else ["test_train.py"]
