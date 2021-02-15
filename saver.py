import pathlib, torch, labml
from labml.experiment import ModelSaver

class TorchSaver(ModelSaver):
    def __init__(self, name: str, model):
        self.name = name
        self.model = model

    def save(self, checkpoint_path: pathlib.Path) -> any:
        state = self.model.state_dict()
        file_name = f"{self.name}.pth"
        torch.save(state, str(checkpoint_path / file_name))
        return file_name

    def load(self, checkpoint_path: pathlib.Path, info: any):
        file_name: str = info
        try:
            sample_param = next(iter(self.model.state_dict().values()))
            device = sample_param.device
        except AttributeError:
            device = torch.device('cpu')
        except StopIteration:
            device = torch.device('cpu')

        state = torch.load(str(checkpoint_path / file_name), map_location=device)

        self.model.load_state_dict(state)
