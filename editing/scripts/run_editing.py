import pyrallis

from editing.src.configs.train_config import TrainConfig
from editing.src.training.trainer import Tex3D


@pyrallis.wrap()
def main(cfg: TrainConfig):
    trainer = Tex3D(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.paint()


if __name__ == '__main__':
    main()
