import os

__all__ = [
    'debug',
    'hyper_params',
    'training_settings'
]

debug = True

hyper_params = {
    
}

HOME = os.path.expanduser("~")

training_settings = {
    'epochs': 1 if debug else 100,
    'training_path': os.path.join(HOME, "storage", "separation", "pt_f_train")

}