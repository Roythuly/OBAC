from utilis.config import Config

default_config = Config({
    "seed": 0,
    "tag": "default",
    "start_steps": 5e3,
    "cuda": True,
    "num_steps": 4000001,
    "save": True,
    
    "env_name": "Ant-v3", 
    "eval": True,
    "eval_numsteps": 10000,
    "eval_times": 5,
    "replay_size": 1000000,

    "algo": "LFP",
    "policy": "Gaussian", 
    "gamma": 0.99, 
    "tau": 0.005,
    "lr": 0.0003,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "batch_size": 512, 
    "updates_per_step": 1,
    "target_update_interval": 1,
    "hidden_size": 512,

    "quantile": 0.9,
    "bc_weight": 0.001,
})