import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Add progress tracking imports
from tqdm import tqdm

def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

def load_images_from_directory(directory):
    dataset = tf.data.Dataset.list_files(directory + "/*.jpg")
    dataset = dataset.map(lambda x: preprocess_image(tf.image.decode_jpeg(tf.io.read_file(x), channels=3)))
    return dataset

# Directories
monet_dir = "./monet_jpg"
photo_dir = "./photo_jpg"

# Load datasets
monet_dataset = load_images_from_directory(monet_dir).batch(1)
photo_dataset = load_images_from_directory(photo_dir).batch(1)

def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, (7, 7), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, (3, 3), strides=2, padding="same", activation="relu")(x)
    for _ in range(6):
        x = residual_block(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(3, (7, 7), padding="same", activation="tanh")(x)
    return models.Model(inputs, outputs)

def residual_block(x):
    inputs = x
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    return layers.Add()([inputs, x])

def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, (4, 4), strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(128, (4, 4), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, (4, 4), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, (4, 4), strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(1, (4, 4), padding="same")(x)
    return models.Model(inputs, outputs)

# Build models
generator_g = build_generator()  # Photo to Monet
generator_f = build_generator()  # Monet to Photo
discriminator_x = build_discriminator()  # Discriminator for Monet images
discriminator_y = build_discriminator()  # Discriminator for Photo images

# Loss functions
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    return real_loss + generated_loss

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

# Optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Training step
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)

        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = tf.reduce_mean(tf.abs(real_x - cycled_x)) + tf.reduce_mean(tf.abs(real_y - cycled_y))

        total_gen_g_loss = gen_g_loss + total_cycle_loss
        total_gen_f_loss = gen_f_loss + total_cycle_loss

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    return {
        "gen_g_loss": total_gen_g_loss,
        "gen_f_loss": total_gen_f_loss,
        "disc_x_loss": disc_x_loss,
        "disc_y_loss": disc_y_loss
    }

# Training loop with progress tracking
EPOCHS = 20
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    epoch_metrics = {
        "gen_g_loss": [],
        "gen_f_loss": [],
        "disc_x_loss": [],
        "disc_y_loss": []
    }

    for real_x, real_y in tqdm(tf.data.Dataset.zip((photo_dataset, monet_dataset)), desc=f"Training Epoch {epoch + 1}"):
        metrics = train_step(real_x, real_y)
        for key, value in metrics.items():
            epoch_metrics[key].append(value.numpy())

    # Calculate average losses for the epoch
    avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
    print(f"Epoch {epoch + 1} Metrics: {avg_metrics}")

# Save models
generator_g.save("generator_g_monet.h5")
generator_f.save("generator_f_photo.h5")

# Test the generator
sample_photo = next(iter(photo_dataset))
monet_style_image = generator_g(sample_photo, training=False)
plt.imshow((monet_style_image[0] + 1) / 2)
plt.show()