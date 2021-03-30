import tensorflow as tf
from official.nlp import optimization


class CompileOnFitKerasModel(tf.keras.Model):
    """
    Keras Tuner expects a function which will return a built AND compiled Keras Model. The limitation of this is that
    scheduling optimizers which set Warmup schedules need to know the number of training steps
    (num_epochs * data_len // batch_size).

    To overcome this, CompileOnFitTfModel is a Keras Model which calls its `compile()` function when `fit()` is called,
    allowing it to use the data length in scheduling optimizers (such as AdamWeightDecay).
    """

    def __init__(self, *args, selected_optimizer=None, optimizer_learning_rate=None, adamw_end_lr=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_optimizer = selected_optimizer
        self.optimizer_learning_rate = optimizer_learning_rate
        self.adamw_end_lr = adamw_end_lr

    def fit(self, *args, **kwargs):
        optimizer = self.selected_optimizer
        if optimizer is None:
            num_train_steps = kwargs["epochs"] * self.get_len(kwargs["x"]) // kwargs["batch_size"]
            optimizer = optimization.create_optimizer(
                init_lr=self.optimizer_learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=int(0.1 * num_train_steps),
                end_lr=self.adamw_end_lr,
                optimizer_type="adamw",
            )
            print("adamw:", num_train_steps, int(0.1 * num_train_steps))

        self.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=tf.metrics.BinaryAccuracy(),
        )
        return super().fit(*args, **kwargs)

    @staticmethod
    def get_len(x):
        if isinstance(x, dict):
            return len(list(x.values())[0])

        return len(x)
