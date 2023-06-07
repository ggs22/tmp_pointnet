import os
from pathlib import Path
import json
import numpy as np
import pytorch_lightning as pl


class SavePredictions(pl.Callback):
    def __init__(self, experiment_output_dir, training_save_frequency=5, validation_save_frequency=1):
        self.experiment_output_dir = experiment_output_dir
        self.training_save_frequency = training_save_frequency
        self.validation_save_frequency = validation_save_frequency
        os.makedirs(experiment_output_dir)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if batch_idx % self.training_save_frequency == 0:
            self._save_keypoint_predictions(**outputs)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx % self.validation_save_frequency == 0:
            self._save_keypoint_predictions(**outputs)

    def _save_keypoint_predictions(self, points, keypoint_predictions, sample_id, stage, **kwargs):
        tx = points.transpose(1, 2).cpu().numpy().squeeze()
        ty = keypoint_predictions.cpu().data.numpy()
        output_path = Path(self.experiment_output_dir).joinpath(f'{sample_id}_{stage}_xyz.txt')
        np.savetxt(fname=str(output_path), X=tx)
        weld_path_dict = dict()
        for weld_path_ix in range(ty.shape[0]):
            weld_path_dict[f'weld_path{weld_path_ix}'] = list()
            for keypoint_ix in range(ty.shape[1]):
                x = float(ty[weld_path_ix, keypoint_ix, :][0])
                y = float(ty[weld_path_ix, keypoint_ix, :][1])
                z = float(ty[weld_path_ix, keypoint_ix, :][2])
                weld_path_dict[f'weld_path{weld_path_ix}'].append([x, y, z])
        output_path = Path(self.experiment_output_dir).joinpath(f'{sample_id}_{stage}_weld_paths.json')
        with open(file=str(output_path), mode='w') as f:
            json.dump(obj=weld_path_dict, fp=f)
