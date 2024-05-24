import datetime


# TODO: If we add support for different weights (pretrained), then we have to add that information to the model name as well
def create_modelname(
    backbone,
    max_epoch,
    batch_size,
    pretrained,
    seed,
    image_size,
    data_flag,
    ssl_method,
    eval_method=None,  # for downstream checkpoints
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pretrained = "pretrained" if pretrained else "not_pretrained"
    modelname = f"{backbone}_{ssl_method}_{max_epoch}_{batch_size}_{seed}_{image_size}_{data_flag}_({pretrained})_{timestamp}"
    if eval_method is not None:
        modelname = f"{eval_method}_" + modelname
    return modelname


def create_ckpt(path, modelname):
    ckpt = path + modelname + ".ckpt"
    return ckpt
