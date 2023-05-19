
from src.experiment.real_train import TrainRealModeling
from src.experiment.synthetic_run_config import SyntheticRunConfig
from src.experiment.synthetic_train import start_training
import os
import pickle as pk
def pretrain_surrogate_model():
    run_config = SyntheticRunConfig(dataset='real',
                                    S=None,
                                    model_name='Tr')
    

    # Setup training
    trainModule = TrainRealModeling(
        run_config,
        batch_size=run_config.batch_size,
        checkpoint_path=run_config.checkpoint_path,
        path_experiment='')
    # store the config of the run
    args_filename = os.path.join(trainModule.checkpoint_path, "param_config.pk")

    with open(args_filename, "wb") as f:
        pk.dump(run_config, f)

    # Start training

    result = trainModule.train_model(run_config.max_iterations,
                                     loss_freq=50,
                                     eval_freq=run_config.eval_freq,
                                     save_freq=run_config.save_freq)

    



def train():

    run_config = SyntheticRunConfig(dataset='proxy',
                                    S=None,
                                    model_name='ArgmaxAR')
    # Setup training
    trainModule = TrainRealModeling(
        run_config,
        batch_size=run_config.batch_size,
        checkpoint_path=run_config.checkpoint_path,
        path_experiment='')
    # store the config of the run
    args_filename = os.path.join(trainModule.checkpoint_path, "param_config.pk")

    with open(args_filename, "wb") as f:
        pk.dump(run_config, f)

    # Start training

    result = trainModule.train_model(run_config.max_iterations,
                                        loss_freq=50,
                                        eval_freq=run_config.eval_freq,
                                        save_freq=run_config.save_freq)


    run_config = SyntheticRunConfig(dataset='proxy',
                                    S=None,
                                    model_name='CNF')
    # Setup training
    trainModule = TrainRealModeling(
        run_config,
        batch_size=run_config.batch_size,
        checkpoint_path=run_config.checkpoint_path,
        path_experiment='')
    # store the config of the run
    args_filename = os.path.join(trainModule.checkpoint_path, "param_config.pk")

    with open(args_filename, "wb") as f:
        pk.dump(run_config, f)

    # Start training

    result = trainModule.train_model(run_config.max_iterations,
                                     loss_freq=50,
                                     eval_freq=run_config.eval_freq,
                                     save_freq=run_config.save_freq)
   
    
    
    


if __name__ == '__main__':
    #pretrain_surrogate_model()
    train()
    
