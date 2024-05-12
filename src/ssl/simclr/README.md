1. WandB log files are stored in `lightning_logs` and `wandb` subdirectories.
2. Tensorboard log files are stored in `tensorboard` subdirectory.
3. Best model checkpoints will be stored inside the `models` folder.
4. To visualize the logs using WandB, you will be prompted to enter your credentials at the beginning of the run. Then you can examine the logs on the official website of WandB.
5. To visualize the logs using Tensorboard, you can run the following command on a separate terminal (assuming you are in the same folder as `main.py`):
```bash
tensorboard --logdir=src/ssl/simclr/tensorboard
```
Then you can have a look at the logs on [http://localhost:6006/](http://localhost:6006/).