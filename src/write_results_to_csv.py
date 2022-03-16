import os
import csv

class factors():
    def __init__(self, args):
        self.factors_dict = {}
        self.factors_dict["alpha"] = args.alpha
        self.factors_dict["beta"] = args.beta
        self.factors_dict["gamma"] = args.gamma
        #self.factors_dict["gamma_intra"] = args.gamma_intra
        #self.factors_dict["gamma_local"] = args.gamma_local
        #self.factors_dict["gamma_global"] = args.gamma_global
        #self.factors_dict["gamma_memory"] = args.gamma
        self.factors_dict["tmp"] = args.temperature
        self.factors_dict["lr"] = args.learning_rate
        self.factors_dict["epoch"] = args.num_train_epochs
        self.factors_dict["seed"] = args.seed
        self.factors_dict["batch"] = args.per_gpu_train_batch_size

def write_results(filename, args, results):

    file_exists = os.path.isfile(filename)
    train_factors = []
    info = factors(args).factors_dict

    for key in sorted(results.keys()):
        info[key] = results[key]

    for key in info:
        train_factors.append(str(key))

    with open(filename, 'a+', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=train_factors)
        if not file_exists:
            writer.writeheader()
        writer.writerow(info)

