"""Train the MNIST network from Class 1 as an Azure ML job.

The same script runs locally (`python train_mnist.py --epochs 1`) and in the cloud.
In Azure ML, MLflow tracking is configured automatically, so every metric we log
shows up in the job's "Metrics" tab and can be compared across runs and sweeps.
"""
import argparse
import os
import time

import keras
import mlflow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--units", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


class MlflowLogger(keras.callbacks.Callback):
    """Send loss/accuracy of every epoch to MLflow (and so to Azure ML)."""

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            mlflow.log_metric(name, float(value), step=epoch)


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    print("Keras", keras.__version__, "backend", keras.backend.backend())
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print("GPUs visible:", gpus)
        mlflow.log_param("gpus", len(gpus))
    except ImportError:
        pass

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255

    model = keras.Sequential([
        keras.Input(shape=(28 * 28,)),
        keras.layers.Dense(args.units, activation="relu"),
        keras.layers.Dropout(args.dropout),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    mlflow.log_params({k: v for k, v in vars(args).items() if k != "output_dir"})

    start = time.perf_counter()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
              validation_split=0.1, callbacks=[MlflowLogger()], verbose=2)
    train_seconds = time.perf_counter() - start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    # The final validation accuracy is what the hyperparameter sweep optimizes.
    val_acc = model.history.history["val_accuracy"][-1]
    mlflow.log_metrics({"final_val_accuracy": val_acc, "test_accuracy": test_acc,
                        "test_loss": test_loss, "train_seconds": train_seconds})
    print(f"val_accuracy={val_acc:.4f} test_accuracy={test_acc:.4f} train_seconds={train_seconds:.1f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, "mnist.keras"))


if __name__ == "__main__":
    main()
