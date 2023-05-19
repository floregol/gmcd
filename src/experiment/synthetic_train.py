import pickle as pk
import os

from src.mutils import PARAM_CONFIG_FILE
from src.train_template import TrainTemplate
from src.experiment.task import TaskSyntheticModeling
from src.model.gmcd.GMCD import GMCD
from src.model.cdm.CDM import CDM
from src.model.cnf.CNFwrapper import CNF
from src.model.argmax.ArgmaxCoupling import ArgmaxCoupling
from src.model.argmax.ArgmaxAR import ArgmaxAr
from src.model.transformer.Transformer import Transformer
# from src.model.gmcd.GMCD import GMCD
from src.datasets.synthetic import SyntheticDataset

# Training class for the synthetic experiment.


class TrainSyntheticModeling(TrainTemplate):
    def __init__(self,
                 runconfig,
                 batch_size,
                 checkpoint_path,
                 path_experiment="",
                 **kwargs):
        self.path_model_prefix = os.path.join(
            "{}_{}".format(runconfig.dataset, runconfig.num_resample),
            "S_{}_K_{}".format(runconfig.S, runconfig.K))
        super().__init__(runconfig,
                         batch_size,
                         checkpoint_path,
                         name_prefix=path_experiment,
                         **kwargs)

    def _create_model(self, runconfig, figure_path):
        if runconfig.model_name == 'CDM':
            model = CDM(run_config=runconfig,
                        dataset_class=SyntheticDataset,
                        figure_path=figure_path)
        elif runconfig.model_name == 'Tr':
            model = Transformer(run_config=runconfig,
                                dataset_class=SyntheticDataset,
                                figure_path=figure_path)
        elif runconfig.model_name == 'GMCD':
            model = GMCD(run_config=runconfig,
                         dataset_class=SyntheticDataset,
                         figure_path=figure_path)
        elif runconfig.model_name == 'ArgmaxAR':
            model = ArgmaxAr(run_config=runconfig,
                             dataset_class=SyntheticDataset,
                             figure_path=figure_path)
        elif runconfig.model_name == 'ArgmaxCoupling':
            model = ArgmaxCoupling(run_config=runconfig,
                                   dataset_class=SyntheticDataset,
                                   figure_path=figure_path)
        elif runconfig.model_name == 'CNF':
            model = CNF(run_config=runconfig,
                                   dataset_class=SyntheticDataset,
                                   figure_path=figure_path)
        return model

    def _create_task(self, runconfig):
        task = TaskSyntheticModeling(self.model,
                                     runconfig,
                                     batch_size=self.batch_size)
        return task


def start_training(runconfig, return_result=False):

    # Setup training
    trainModule = TrainSyntheticModeling(
        runconfig,
        batch_size=runconfig.batch_size,
        checkpoint_path=runconfig.checkpoint_path,
        path_experiment='')
    # store the config of the run
    args_filename = os.path.join(trainModule.checkpoint_path,
                                 PARAM_CONFIG_FILE)
    if runconfig.model_name!='CNF':

        with open(args_filename, "wb") as f:
            pk.dump(runconfig, f)

    # Start training

    result = trainModule.train_model(runconfig.max_iterations,
                                     loss_freq=50,
                                     eval_freq=runconfig.eval_freq,
                                     save_freq=runconfig.save_freq)

    # Cleaning up the checkpoint directory afterwards if selected

    if return_result:
        return result
