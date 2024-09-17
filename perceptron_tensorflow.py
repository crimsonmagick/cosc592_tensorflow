import tensorflow as tf
import numpy as np

# Set random seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

inputs = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

expected_outputs = np.array([
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,))
])

# Use SGD optimizer with a higher learning rate
optimizer = tf.keras.optimizers.SGD(learning_rate=0.3)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

class_weight = {0: 3, 1: 1}
model.fit(inputs, expected_outputs, epochs=100, class_weight=class_weight, verbose=0)

loss, accuracy = model.evaluate(inputs, expected_outputs, verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')

outputs = model.predict(inputs)

print("Input\tExpected\tOutput")
for i in range(len(inputs)):
    print(f"{inputs[i]}\t{expected_outputs[i]}\t{outputs[i]}")
