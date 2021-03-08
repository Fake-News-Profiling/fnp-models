from kerastuner.tuners import BayesianOptimization


class BayesianOptimizationTunerWithFitHyperParameters(BayesianOptimization):
    """
    BayesianOptimization keras Tuner which passes batch_size and epochs hyper-parameters to the fit method of the
    keras Model
    """

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        fit_kwargs.setdefault("batch_size", trial.hyperparameters.get("batch_size"))
        fit_kwargs.setdefault("epochs", trial.hyperparameters.get("epochs"))
        super().run_trial(trial, *fit_args, **fit_kwargs)
