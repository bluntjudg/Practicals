import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam


def main():
    tf.keras.utils.set_random_seed(42)
    (real_images, _), _ = mnist.load_data()
    real_images = (real_images.astype("float32") - 127.5) / 127.5
    real_images = real_images.reshape(-1, 784)
    print("Real data shape:", real_images.shape)

    noise_dimension = 100
    generator = Sequential(
        [
            Input(shape=(noise_dimension,)),
            Dense(256),
            LeakyReLU(negative_slope=0.2),
            Dense(512),
            LeakyReLU(negative_slope=0.2),
            Dense(1024),
            LeakyReLU(negative_slope=0.2),
            Dense(784, activation="tanh"),
        ],
        name="generator",
    )
    discriminator = Sequential(
        [
            Input(shape=(784,)),
            Dense(512),
            LeakyReLU(negative_slope=0.2),
            Dense(256),
            LeakyReLU(negative_slope=0.2),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=["accuracy"],
    )

    discriminator.trainable = False
    noise = Input(shape=(noise_dimension,))
    generated_image = generator(noise)
    validity = discriminator(generated_image)
    gan = Model(noise, validity, name="gan")
    gan.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    )

    steps = 3000
    batch_size = 128
    for step in range(1, steps + 1):
        real_batch = real_images[
            np.random.randint(0, len(real_images), batch_size)
        ]
        noise_batch = np.random.normal(
            0,
            1,
            (batch_size, noise_dimension),
        ).astype("float32")
        fake_batch = generator.predict(noise_batch, verbose=0)

        discriminator.trainable = True
        real_loss = discriminator.train_on_batch(
            real_batch,
            np.ones((batch_size, 1), dtype="float32"),
        )
        fake_loss = discriminator.train_on_batch(
            fake_batch,
            np.zeros((batch_size, 1), dtype="float32"),
        )
        discriminator.trainable = False
        discriminator_loss = 0.5 * (real_loss[0] + fake_loss[0])

        generator_loss = gan.train_on_batch(
            noise_batch,
            np.ones((batch_size, 1), dtype="float32"),
        )
        if step % 500 == 0:
            print(
                f"Step {step}/{steps} "
                f"D loss: {discriminator_loss:.4f} "
                f"G loss: {float(np.squeeze(generator_loss)):.4f}"
            )

    sample_count = 1000
    noise_batch = np.random.normal(
        0,
        1,
        (sample_count, noise_dimension),
    ).astype("float32")
    synthetic = generator.predict(noise_batch, verbose=0)
    real_sample = real_images[
        np.random.randint(0, len(real_images), sample_count)
    ]

    real_scores = discriminator.predict(real_sample, verbose=0)
    synthetic_scores = discriminator.predict(synthetic, verbose=0)
    print("Real data discriminator score:", float(real_scores.mean()))
    print("Synthetic data discriminator score:", float(synthetic_scores.mean()))
    print("Real mean and standard deviation:", real_sample.mean(), real_sample.std())
    print("Synthetic mean and standard deviation:", synthetic.mean(), synthetic.std())
    print("Real minimum and maximum:", real_sample.min(), real_sample.max())
    print("Synthetic minimum and maximum:", synthetic.min(), synthetic.max())


if __name__ == "__main__":
    main()
