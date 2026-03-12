import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Load Data
# We use a small batch size of 8 because your dataset is small
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=8
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=8
)

# 2. Build the "Smart" Brain
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    
    # Randomly flip and rotate images so the AI learns better
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # This stops it from over-learning dry road patterns
    layers.Dense(1, activation='sigmoid')
])

# Use a slow, precise learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# 3. Train for 40 epochs
print("Training started... this will make the model much smarter.")
model.fit(train_ds, validation_data=val_ds, epochs=40)

# 4. Save the file
model.save('water_model.h5')
print("Model saved as water_model.h5!")