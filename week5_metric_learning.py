from src.metric_learning.build_dataset import generate_train_crops
from src.metric_learning.train import train
from pathlib import Path

"""
generate_train_crops(root=str(Path.joinpath(Path(__file__).parent, './aic19-track1-mtmc-train/train')),
                    save_path=str(Path.joinpath(Path(__file__).parent, './aic19_database')),
                    train_seqs=['S01', 'S04'],
                    val_seqs=['S03'])
"""

train(data_path=str(Path.joinpath(Path(__file__).parent, './aic19_database')),
      save_path=str(Path.joinpath(Path(__file__).parent, './checkpoints')),
      log_path=str(Path.joinpath(Path(__file__).parent, './checkpoints')))