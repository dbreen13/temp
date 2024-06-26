import json
import os
from time import time, perf_counter
from pathlib import Path
from typing import List, Union
from functools import partial

import git
import yaml
import numpy as np
# from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.suggest.bayesopt import BayesOptSearch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torchvision.models import resnet18 

from tddl.dbs import find_error_given_c
from tddl.dbs import undo_factorize_rank_1
from tddl.dbs import get_errors_given_c
from tddl.data.loaders import fetch_loaders
from tddl.models.resnet_torch import get_resnet_from_torch
from tddl.models.cnn import GaripovNet, JaderbergNet
from tddl.trainer import Trainer
from tddl.models.wrn import WideResNet
from tddl.models.pa_resnet import PA_ResNet18
from tddl.models.pa_resnet_lr import low_rank_resnet18
from tddl.utils.random import set_seed
from tddl.utils.hardware import select_hardware
from tddl.models.utils import count_parameters
from tddl.utils.checks import check_paths
from tddl.factorizations import number_layers
from tddl.factorizations import listify_numbered_layers
from tddl.factorizations import factorize_network
from tddl.factorizations import factorize_network_different_ranks
from tddl.factorizations import list_errors
from tddl.utils.typecast import typecast

import logging
import time as timers

import tensorly as tl

import typer

app = typer.Typer()


tl.set_backend('pytorch')

DATASET_MODEL_PARAMETERS = { # TODO: write this as config file
    'fmnist':{
        'num_classes': 10,
        'in_channels': 1,
    },
    'cifar10':{
        'num_classes': 10,
        'in_channels': 3,
    }
}


@app.command()
@typecast
def train(
    batch: int = 256,
    epochs: int = 200,
    logdir: Path = Path("/home/dbreen/Documents/tddl/bigdata"),
    lr: float = 0.1,
    gamma: float = 0.1,
    dropout: float = None,
    model_name: str = "parn",
    depth: int = 18,
    width: int = 10,
    cpu: int = None,
    data_workers: int = 1,
    seed: int = None,
    data_dir: Path = Path("/home/dbreen/Documents/tddl/bigdata"),
    cuda: str = None,
    milestones: List[int] = None,
    optimizer: str = None,
    weight_decay: float = 1e-4,
    t: float = None,
    dataset: str ='fmnist',
    momentum: float = 0.9,
    index: int=1,
    **kwargs
) -> None:

    logdir, data_dir = check_paths(logdir, data_dir)

    select_hardware(cuda, cpu)
    if cpu is not None:
        data_workers = int(cpu) # max(int(cpu)-1, 1)
    
    if not logdir.is_dir():
        raise FileNotFoundError("{0} folder does not exist!".format(logdir))
    if t is None:
        t = round(time())
    if seed is None:
        seed = t
    set_seed(seed)
    logging_path2=f"baseline-{model_name}-{dataset}/runnr{index}"
    MODEL_NAME = f"{model_name}_{depth}_d{dropout}_{batch}_{optimizer}_l{lr}_g{gamma}_w{weight_decay}_s{seed == t}"
    logdir = logdir.joinpath(logging_path2, MODEL_NAME)
    print(logdir)
    save = {
        "save_every_epoch": None,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": "cnn"
    }

    train_loader, valid_loader, test_loader = fetch_loaders(
        dataset=dataset,
        path=data_dir,
        batch_size=batch,
        data_workers=data_workers,
        valid_size=5000,
    )
    
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))
    
    
    logger=logging.getLogger('Lekkertesten')
    #create a fh
    fh=logging.FileHandler('testen.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
        
    model_parameters = DATASET_MODEL_PARAMETERS.get(dataset)
    # if model_name == 'wrn':
    #     model = WideResNet(
    #         depth=depth,
    #         num_classes=num_classes,
    #         widen_factor=width,
    #         dropRate=dropout,
    #     ).cuda()
    #     default_milestones = [100, 150, 225]
    # elif model_name == "parn":
    #     model = PA_ResNet18(
    #         num_classes=num_classes, 
    #         nc=1,
    #     ).cuda()
    #     default_milestones = [100, 150]
    if model_name == "rn18":
        model = get_resnet_from_torch(
            in_channels=model_parameters['in_channels'],
            num_classes=model_parameters['num_classes'],
        ).cuda()
        # default_milestones = [100, 150]
    elif model_name == "gar":
        model = GaripovNet(
            in_channels=model_parameters['in_channels'],
            num_classes=model_parameters['num_classes'],
        ).cuda()
    elif model_name == "jad":
        model = JaderbergNet(
            in_channels=model_parameters['in_channels'],
            num_classes=model_parameters['num_classes'],
        ).cuda()
    else:
        raise NotImplementedError

    n_param = count_parameters(model)
    with open(logdir.joinpath('n_param.json'), 'w') as f:
        json.dump(n_param, f)

    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) if milestones is not None else None
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    trainer = Trainer(
        train_loader, valid_loader, test_loader,
        model, optimizer, writer, scheduler=scheduler, save=save
    )
    start_training = perf_counter()
    timers.sleep(120)
    logger.info('start-bas%s-mdl%s-ind%s' %(dataset,model_name,index))
    results = trainer.train(epochs=epochs)
    logger.info('end-bas%s-mdl%s-ind%s' %(dataset,model_name,index))
    end_training = perf_counter()
    training_time = start_training - end_training
    results['training_time'] = training_time
    results['training_time_per_epoch'] = training_time/epochs
    writer.add_scalar("Timing/training", training_time)
    writer.add_scalar("Timing/training_per_epoch", training_time/epochs)

    # this uses final model, need to use best model
    # test_acc, test_loss = trainer.test(loader=test_loader)
    # load best model
    trainer.model = torch.load(Path(trainer.save_location) / f"{trainer.save_model_name}_best.pth")
    # test best model
    start_inference_time = perf_counter()
    test_acc, test_loss = trainer.test(loader=test_loader)
    inference_time = start_inference_time - perf_counter()
    # register test results
    results['test_acc'] = test_acc
    results['test_loss'] = test_loss
    results['n_param'] = n_param
    results['model_name'] = MODEL_NAME
    results['inference_time'] = inference_time
    writer.add_scalar("Timing/inference", inference_time)

    with open(logdir.joinpath('results.json'), 'w') as f:
        json.dump(results, f)

    writer.close()


@app.command()
@typecast
def decompose(
    layers: Union[int, List[int]],
    baseline_path: Path = Path("/home/dbreen/Documents/tddl/pretrained/f_mnist/parn_18_d0.5_256_sgd_l0.1_g0.1/1629473591/cnn_best"),
    factorized_path: Path = None,
    factorization: str = 'tucker',
    decompose_weights: bool = True,
    td_init: float = None, # 0.02
    rank: Union[float, List[float]] = 0.5, #: float or list
    different_ranks: bool = False,
    cache: bool = True,                         #TODO
    factorize_rank_1: bool = True,
    bound_search: bool = False,                 #TODO
    error: float = 0.5,
    error_max: float = 1.0,
    error_min: float = 0.0,
    epochs: int = 200,
    lr: float = 0.1,
    logdir: Path = Path("/home/dbreen/Documents/tddl/bigdata"),
    # freeze_parameters: bool = False
    batch: int = 256,
    gamma: float = 0,
    model_name: str = "parn",
    seed: int = None,
    data_workers: int = 2,
    data_dir: Path = Path("/home/dbreen/Documents/tddl/bigdata"),
    cuda: str = None,
    cpu: str = None,
    checkpoint_dir: str = None,
    config: str = None, #raytune config
    return_error: bool = False,
    weight_decay: float = 0,
    optimizer: str = 'adam',
    t: float = None,
    dataset: str = 'fmnist',
    momentum: float = 0.9,
    index: int=1,
    **kwargs,
) -> None:
    
    if type(layers) == int:
        layers = [layers]
    if all(isinstance(elem, list) for elem in layers): 
        layers=layers[0]
        
    print('hier ben ik')    
    logdir, data_dir, baseline_path = check_paths(
        logdir, data_dir, baseline_path
    )

    select_hardware(cuda, cpu)
    
    if cpu is not None:
        data_workers = int(cpu) # max(int(cpu)-1, 1)
    
    if not decompose_weights:
        if td_init is None:
            print(10*"!", "td_init not set", 10*"!") # TODO logging, not printing
            td_init=0.02
            print(10*"!", f"set td_init to: {td_init}", 10*"!")

    tuning = False
    if config is not None:
        lr = config['lr']
        factorization = config['fact']
        rank = config['rank']
        gamma = config['gamma']
        decompose_weights = config['td']
        tuning = True

    if not logdir.is_dir():
        raise FileNotFoundError("{0} folder does not exist!".format(logdir))
    td = "td" if not decompose_weights else "lr"
    
    if t is None:
        t = round(time())
    if seed is None:
        seed = t
    set_seed(seed)

    MODEL_NAME = f"{model_name}-{td}-{layers}-{factorization}-{rank}-d{str(decompose_weights)}-i{td_init}_bn_{batch}_sgd_l{lr}_g{gamma}_s{seed == t}"
    logging_path=f"fact-{factorization}-r{rank}-lay{layers}/runnr{index}"
    logdir = logdir.joinpath(logging_path, MODEL_NAME)
    save = {
        "save_every_epoch": None,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": f"fact_model"
    }
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))

    output = None
    if factorized_path is not None:
        model = torch.load(factorized_path)
    else:
        compression_log = {}
        model = torch.load(baseline_path)
        start_compression_time = perf_counter()
        if different_ranks:
            if bound_search:
                start_get_bounds = perf_counter()
                errors = get_errors_given_c(model, layers, rank)
                get_bounds_time = start_get_bounds - perf_counter()
                compression_log['get_bounds_time'] = get_bounds_time
                writer.add_scalar("Timing/get_bounds_time", get_bounds_time)

                # overwrite input values of function `decompose()`
                error = errors.mean()
                max_error = errors.max()
                min_error = errors.min()
            if type(rank) is not list:
                baseline_count = count_parameters(model)
                numbered_layers = number_layers(model)
                listed_layers = listify_numbered_layers(numbered_layers, layer_nrs=layers)
                
                start_find_ranks = perf_counter()
                rank, c, error = find_error_given_c(
                    listed_layers, 
                    desired_c = rank, 
                    baseline_count=baseline_count,
                    error = error,
                    max_error=max_error, 
                    min_error=min_error,
                )
                finding_ranks_time = start_find_ranks - perf_counter()
                compression_log['finding_ranks_time'] = finding_ranks_time
                writer.add_scalar("Timing/finding_ranks_time", finding_ranks_time)
            

            if not factorize_rank_1:
                # TODO: log which layers have relative rank 1.0 and end up not being factorized
                print(f"layer before filtering 1.0: {layers}")
                layers, rank = undo_factorize_rank_1(layers,rank)
                print(f"layer after filtering 1.0: {layers}")
            
            start_factorize = perf_counter()
            factorize_network_different_ranks(
                model, 
                layers, 
                rank, 
                factorization=factorization,
                decompose_weights=decompose_weights,
                init_std=td_init,
                return_error=return_error,
            )
            factorizing_time = start_factorize - perf_counter()


        else:
            logger=logging.getLogger('Lekkertesten')
            #create a fh
            fh=logging.FileHandler('testen.log')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            #logger.info('Ik begin nu met factorizen van dec-%s-r%s-%s' %(factorization, rank, layers))
            start_factorize = perf_counter()
            output = factorize_network(
                model,
                layers=layers,
                factorization=factorization,
                rank=float(rank),
                decompose_weights=decompose_weights,
                init_std=td_init,
                return_error=return_error,
            )
            factorizing_time = start_factorize - perf_counter()

            #logger.info('Ik eindig nu met factorizen van dec-%s-r%s-%s' %(factorization, rank, layers))
        compression_time = start_compression_time - perf_counter()
        compression_log['compression_time'] = compression_time
        writer.add_scalar("Timing/compression_time", compression_time)
        compression_log['factorizing_time'] = factorizing_time
        writer.add_scalar("Timing/factorizing_time", factorizing_time)
        compression_log['rank'] = rank
        with open(logdir.joinpath('compression_log.json'), 'w') as f:
            json.dump(compression_log, f)

    model.cuda() # needed for factorized training
    # Save the factorized model to the current logdir, also if it is loaded from another run
    torch.save(model, logdir / "model_after_fact.pth")

    
    # TODO modules are in here, they are not serializable
    # with open(logdir.joinpath('factorization.json'), 'w') as f:
    #     json.dump(output, f)
    # print(output)

    if output is not None:
        errors = list_errors(output, layers)
        with open(logdir.joinpath('erros.json'), 'w') as f:
            json.dump(errors, f)

    n_param = count_parameters(model) # TODO: is gradients check really necessary? Does this mess up tt? 
    with open(logdir.joinpath('n_param.json'), 'w') as f:
        json.dump(n_param, f)

    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # TODO check schedulers
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) if gamma else None
    # if not decompose_weights:
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=gamma)
    scheduler = None

    if checkpoint_dir:
        model_state, optimizer_state, scheduler_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)

    train_loader, valid_loader, test_loader = fetch_loaders(
        dataset=dataset,
        path=data_dir,
        batch_size=batch,
        data_workers=data_workers,
        valid_size=5000,
    )

    # TODO Session not detected. You should not be calling this function outside `tune.run` or while using the class API.
    trainer = Trainer(
        train_loader, valid_loader, test_loader,
        model, optimizer, writer, 
        scheduler=scheduler, save=save, tuning=tuning)
    
    if decompose_weights:
        #logger.info('Ik begin nu met before training van dec-%s-r%s-%s' %(factorization, rank, layers))
        train_acc, train_loss = trainer.test(loader="train")
        writer.add_scalar("Accuracy/before_finetuning/train", train_acc)
        writer.add_scalar("Loss/before_finetuning/train", train_loss)
        valid_acc, valid_loss = trainer.test(loader="valid")
        writer.add_scalar("Accuracy/before_finetuning/valid", valid_acc)
        writer.add_scalar("Loss/before_finetuning/valid", valid_loss)
        test_acc, test_loss = trainer.test(loader="test")
        writer.add_scalar("Accuracy/before_finetuning/test", test_acc)
        writer.add_scalar("Loss/before_finetuning/test", test_loss)
        results_before_training = {
            "train_acc": train_acc,
            "train_loss": train_loss,
            "valid_acc": valid_acc,
            "valid_loss": valid_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
        }
        with open(logdir.joinpath('results_before_training.json'), 'w') as f:
            json.dump(results_before_training, f)
    #logger.info('Ik eindig nu met before training van dec-%s-r%s-%s' %(factorization, rank, layers))      
    start_training = perf_counter()
    timers.sleep(120)
    logger.info('start-dec-%s-r%s-lay%s-ind%s' %(factorization, rank, layers,index)) 
    results = trainer.train(epochs=epochs)
    logger.info('end-dec-%s-r%s-lay%s-ind%s' %(factorization, rank, layers,index))
    timers.sleep(120)
    end_training = perf_counter()
    training_time = start_training - end_training
    results['training_time'] = training_time
    writer.add_scalar("Timing/training", training_time)
    results['training_time_per_epoch'] = training_time/epochs
    writer.add_scalar("Timing/training_per_epoch", training_time/epochs)
    
    # load best model
    trainer.model = torch.load(Path(trainer.save_location) / f"{trainer.save_model_name}_best.pth")
    # test best model
    start_inference_time = perf_counter()
    test_acc, test_loss = trainer.test(loader=test_loader)
    inference_time = start_inference_time - perf_counter()
    # register test results
    results['test_acc'] = test_acc
    results['test_loss'] = test_loss
    results['model_name'] = MODEL_NAME
    results['n_param_fact'] = n_param
    results['inference_time'] = inference_time
    writer.add_scalar("Timing/inference", inference_time)
    # if decompose_weights:
    #     results['train_acc_before_ft'] = train_acc
    #     results['valid_acc_before_ft'] = valid_acc
    with open(logdir.joinpath('results.json'), 'w') as f:
        json.dump(results, f)

    writer.close()


# Config needs to be first positional argument of trainable_function for RayTune
def tune_decompose(config, checkpoint_dir, data_dir, *args, **kwargs):
    # print({**kwargs})
    # if config['fact']:
    #     factorization=config['fact']
    decompose(*args, config=config, checkpoint_dir=checkpoint_dir, data_dir=data_dir, **kwargs)


@app.command()
@typecast
def hype(
    layers: List[int],
    lr: float = 1.e-1,
    runtype: str ='decompose',
    num_samples: int = 10,
    max_epochs: int = 10,
    gpus_per_trial: int = 1,
    cpus_per_trial: int = 4,
    baseline_path: str = "/home/dbreen/Documents/tddl/pretrained/f_mnist/parn_18_d0.5_256_sgd_l0.1_g0.1/1629473591/cnn_best",
    factorization: str = 'tucker',
    decompose_weights: bool = True,
    td_init: float = 0.02,
    rank = 0.5,
    epochs: int = 200,
    logdir: Path = Path("/home/dbreen/Documents/tddl/pretrained/f_mnist"),
    # freeze_parameters: bool = False,
    batch: int = 256,
    gamma: float = 0,
    model_name: str = "parn",
    seed: int = None,
    data_workers: int = 1,
    data_dir: Path = Path("/home/dbreen/Documents/tddl/bigdata"),
    cuda: str = None,
    cpu: str = None,
    checkpoint_dir: str = None,
    search_type: str = 'grid',
    metric: str = 'loss',
    mode: str = 'min',
    grace_period: int = 1,
    **kwargs
) -> None:
    """
    hyperparameter tuning

    input:
        lr: list of [min, max] learning rate
        runtype: string in ['train', 'decompose']
        search_type: string in ['grid', 'loguniform'] # TODO not implemented yet, will be difficult when more hyperparameters are tuned.
        metric: string in ['loss', 'accuracy']
        mode: string in ['min', 'max']
        rank: float (frational rank) or str (0.25, 0.5, 0.75)
        factorization: all ('tucker', 'cp', 'tt')
    """
    
    logdir, data_dir = check_paths(logdir, data_dir)

    select_hardware(cuda, cpu)
    if cpu is not None:
        data_workers = int(cpu) # max(int(cpu)-1, 1)

    # lr = np.arange(lr_min, lr_max, 1)

    lrs = lr if type(lr) == list else [lr]
    # lr = [0.1**x for x in [1,2,3,4,5]] # 0 did not work
    # np.power(0.1, np.array([0,1,2,3,4,5]), dtype=float)
    gammas = gamma if type(gamma) == list else [gamma]
    ranks = rank if type(rank) == list else [rank]
    # rank_list = [0.25, 0.5, 0.75] if rank == "all" else [float(rank)]
    facts = factorization if type(factorization) == list else [factorization]
    
    # TODO this is hacky, check if typer can be used to call functions
    if type(decompose_weights) == str:
        decompose_weights = bool(int(decompose_weights))
    # tds: should you initialize with td (1) or randomly (0)
    tds = decompose_weights if type(decompose_weights) == list else [bool(decompose_weights)]
    
    config = {
        # "lr": tune.loguniform(lr_min, lr_max),
        "lr": tune.grid_search(lrs), # array([1.e+00, 1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05])
        "gamma": tune.grid_search(gammas),
        "rank": tune.grid_search(ranks),
        "fact": tune.grid_search(facts),
        "td": tune.grid_search(tds),
    }

    scheduler = ASHAScheduler(
        metric=metric, # "loss"
        mode=mode, # "min"
        max_t=max_epochs,
        grace_period=grace_period,     #TODO: Only stop trials at least this old in time. Unclear what definition of time unit is
        reduction_factor= 2,  #TODO: what does reduction_factor do? Halving, check ASHA 
    )

    reporter = CLIReporter(
        # parameter_columns=["lr", "gamma", "fact", "rank"],
        metric_columns=["loss", "accuracy", "training_iteration"],
    )
    
    if runtype == 'decompose':
        train_func = tune_decompose
    elif runtype == 'train':
        train_func = train
    else:
        raise NotImplementedError

    checkpoint_dir = checkpoint_dir if checkpoint_dir else logdir / "check_dir"

    # algo = BayesOptSearch(
    #     metric=metric,
    #     mode=mode,
    #     utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
    #     random_search_steps=30,
    # )
    
    # algo = ConcurrencyLimiter(algo, max_concurrent=10)

    result = tune.run(
        partial(train_func, 
            layers=layers,
            baseline_path=baseline_path,
            # factorization=config['fact'],
            decompose_weights=decompose_weights,
            td_init=td_init,
            # rank=rank,
            epochs=epochs,
            logdir=logdir,
            # freeze_parameters: bool = False,
            batch=batch,
            # gamma=gamma,
            model_name=model_name,
            seed=seed,
            data_workers=data_workers,
            data_dir=data_dir,
            # cuda=cuda,
            # cpu=cpu,
            checkpoint_dir= checkpoint_dir,
            **kwargs,
        ),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        # search_alg=algo,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=logdir / "raytune",
        # name="MEHBLAH", # default: DEFAULT_2021-11-16_13-47-23
        # trial_name_creator=trial_name_string # trial_name_string is function
    )

    best_trial = result.get_best_trial(metric, mode, "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # # Restored previous trial from the given checkpoint
    # tune.run(
    #     "PG",
    #     name="RestoredExp", # The name can be different.
    #     stop={"training_iteration": 10}, # train 5 more iterations than previous
    #     restore="~/ray_results/Original/PG_<xxx>/checkpoint_5/checkpoint-5",
    #     config={"env": "CartPole-v0"},
    # )


@app.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
def main(
    ctx: typer.Context,
    config_path: Path = Path("/home/dbreen/Documents/tddl/"),
    ) -> None:
    
    #configure a logger
    # Create our demo logger
    logger = logging.getLogger('Lekkertesten')
    # Set a log level for the logger
    logger.setLevel(logging.INFO)
    # Create a console handler 
    handler = logging.StreamHandler()
    # Set INFO level for handler
    handler.setLevel(logging.INFO)
    # Create a message format that matches earlier example
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Add our format to our handler
    handler.setFormatter(formatter)
    # Add our handler to our logger
    logger.addHandler(handler)
  
    
    config_data = {}
    # load yaml / alternative https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python
    config_data = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    # typer.echo(f"Config file: {config_data}")
    for item in ctx.args:
        config_data.update([item.strip("--").split('=')])
    # typer.echo(f"Config, modified with inputs: {config_data}")
    
    t = round(time())
    seed = config_data.get("seed")
    if seed is None:
        seed = t
    config_data.update({"seed":seed})
    config_data.update({"t":t})

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config_data.update({"hexsha":sha})


    # write yaml to logdir folder
    if config_data.get('factorization') is None:
        config_log = Path(config_data['logdir']) / f"baseline-{config_data['model_name']}-{config_data['dataset']}"/ f"runnr{config_data['index']}"/'config.yml'
    else: 
        config_log = Path(config_data['logdir']) / f"fact-{config_data['factorization']}-r{config_data['rank']}-lay{config_data['layers']}" / f"runnr{config_data['index']}"/'config.yml'
   
    os.makedirs(os.path.dirname(config_log), exist_ok=True)
    with open(config_log, 'w', encoding='utf8') as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False, allow_unicode=True)

    # config_data_log = yaml.load(Path("/home/jetzeschuurman/gitProjects/phd/tddl/tmp/1637934961/config.yml").read_text(), Loader=yaml.Loader)
    
    if config_data.get('tune'):
        hype(**config_data)

        #TODO continue from hyperparameter tuning and train best model.

    elif config_data.get('factorization') is None:
        train(**config_data) # TODO check if function can be called with typer, such that path_string is interpreted as Path instance not str
    else:
        decompose(**config_data)


if __name__ == "__main__":
    app()
