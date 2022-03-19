from pytorch_lightning.callbacks import RichProgressBar as _RichProgressBar


class RichProgressBar(_RichProgressBar):

    def _update_metrics(self, trainer, pl_module) -> None:
        metrics = self.get_metrics(trainer, pl_module)
        metrics['global_step'] = trainer.global_step
        if self._metric_component:
            self._metric_component.update(metrics)
