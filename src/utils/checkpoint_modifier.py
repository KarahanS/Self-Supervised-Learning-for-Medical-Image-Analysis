import torch
from omegaconf import DictConfig, OmegaConf
from src.ssl.methods import METHODS
from src.args.pretrain import parse_cfg, _N_CLASSES_MEDMNIST


# Keep only the backbones - remove the rest (classifier, projector, etc.)

class CheckpointModifier:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        OmegaConf.set_struct(self.cfg, False)
        self.cfg = parse_cfg(self.cfg)
        self.ckpt_path = self.cfg.load.path

    def duplicate_ckpt(self):
        """Duplicates the checkpoint file."""
        try:
            ckpt = torch.load(self.ckpt_path)
        except FileNotFoundError:
            print(f"Error: Checkpoint file '{self.ckpt_path}' not found.")
            return None

        new_ckpt_path = self.ckpt_path.replace('.ckpt', '_modified.ckpt')
        torch.save(ckpt, new_ckpt_path)
        print(f"All changes will be saved to: {new_ckpt_path}")
        return new_ckpt_path
    
    def remove_keys_starting_with(self, state_dict, prefix):
        keys_to_delete = [key for key in state_dict.keys() if key.startswith(prefix)]
        for key in keys_to_delete:
            del state_dict[key]
        return state_dict
    
    def rename_keys_starting_with(self, state_dict, prefix, new_prefix):
        keys_to_rename = [key for key in state_dict.keys() if key.startswith(prefix)]
        for key in keys_to_rename:
            new_key = key.replace(prefix, new_prefix)
            state_dict[new_key] = state_dict.pop(key)
        return state_dict
    
    def reset_classifier(self, ckpt_path: str):
        """Resets the classifier head in the checkpoint state dictionary."""
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']


        if self.cfg.method in ["simclr", "byol", "dino", "vicreg"]:
            self.remove_keys_starting_with(state_dict, 'online_classifier')
            
        elif self.cfg.method == "mocov3":  # selfmmsup
            self.remove_keys_starting_with(state_dict, 'head.predictor')

        else:
            raise NotImplementedError(f"Method {self.cfg.method} not implemented yet.")
        
        torch.save(ckpt, ckpt_path)
        print(f"Modified checkpoint saved to: {ckpt_path}")


    def reset_projector(self, ckpt_path: str):
        """Resets the projector head in the checkpoint state dictionary."""
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']

        if self.cfg.method == "simclr":  # projection head will be initialized randomly
            # if you don't delete these keys, it won't give an error, but a warning will be printed:
            # Found keys that are not in the model state dict but in the checkpoint: ...
            self.remove_keys_starting_with(state_dict, 'projection_head')

        elif self.cfg.method == "byol": # all of the heads will be initialized randomly
            # 2 backbones: backbone and momentum_backbone
            self.remove_keys_starting_with(state_dict, 'projection_head')

            self.remove_keys_starting_with(state_dict, 'prediction_head')
            self.remove_keys_starting_with(state_dict, 'teacher_projection')
            
            self.rename_keys_starting_with(state_dict, 'teacher_backbone', 'momentum_backbone') # rename
            # classifier, projector, momentum_projector, predictor will be initialized randomly
            
        elif self.cfg.method == "dino":
            # 2 backbones: student_backbone and momentum_backbone
            self.remove_keys_starting_with(state_dict, 'projection_head')
            self.remove_keys_starting_with(state_dict, 'student_projection')
            self.remove_keys_starting_with(state_dict, 'criterion')
           
            self.rename_keys_starting_with(state_dict, 'student_backbone', 'momentum_backbone')
            # classifier, head, momentum_head, dino_loss_func will be initialized randomly
        
        elif self.cfg.method == "vicreg":
            self.remove_keys_starting_with(state_dict, 'projection_head')
        
        elif self.cfg.method == "mocov3":
            # again, take only the backbone
            self.remove_keys_starting_with(state_dict, 'data_preprocessor')
            self.remove_keys_starting_with(state_dict, 'neck')
            self.remove_keys_starting_with(state_dict, 'momentum_encoder.module.1')
            self.remove_keys_starting_with(state_dict, 'momentum_encoder.steps')
            
            self.rename_keys_starting_with(state_dict, 'momentum_encoder.module.0', 'momentum_backbone')
            
        else:
            raise NotImplementedError(f"Method {self.cfg.method} not implemented yet.")
        
        torch.save(ckpt, ckpt_path)
        print(f"Modified checkpoint saved to: {ckpt_path}")
        
    def add_pytorch_lightning_version(self, ckpt_path: str):
        """Adds the pytorch-lightning version to the checkpoint."""
        ckpt = torch.load(ckpt_path)
        if 'pytorch-lightning_version' not in ckpt:
            ckpt['pytorch-lightning_version'] = '2.0.1.post0' # dummy version
            torch.save(ckpt, ckpt_path)
            print(f"Modified checkpoint saved to: {ckpt_path}")


    def process_checkpoint(self):
        """Main function to process the checkpoint by duplicating and resetting heads."""
        new_ckpt_path = self.duplicate_ckpt()
        if new_ckpt_path:
            self.reset_classifier(new_ckpt_path)
            self.reset_projector(new_ckpt_path)
            self.add_pytorch_lightning_version(new_ckpt_path)


