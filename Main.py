import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding, RepeatVector, Layer, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- Attention Layer ---
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# --- Data Preprocessing ---
def preprocess_data(data):
    data['DATE_STAMP'] = pd.to_datetime(data['DATE_STAMP'])
    data.sort_values(by=['OBJT_ID', 'DATE_STAMP'], inplace=True)
    data = data.groupby('OBJT_ID').apply(lambda group: group.fillna(method='bfill')).reset_index(drop=True)
    return data

# --- Sequence Creation ---
def create_sequences(data, obj_ids, features, target_col, window_size, forecast_horizon):
    X, y, ids = [], [], []
    temperature_scalers = {}

    for obj_id in obj_ids:
        obj_data = data[data['OBJT_ID'] == obj_id]
        if len(obj_data) < window_size + forecast_horizon:
            continue

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        scaled_features = feature_scaler.fit_transform(obj_data[features])
        scaled_target = target_scaler.fit_transform(obj_data[[target_col]])

        temperature_scalers[obj_id] = target_scaler

        for i in range(len(obj_data) - window_size - forecast_horizon):
            X.append(scaled_features[i:i+window_size])
            y.append(scaled_target[i+window_size:i+window_size+forecast_horizon])
            ids.append(obj_id)

    return np.array(X), np.array(y), np.array(ids), temperature_scalers

# --- Model Building ---
def build_model(window_size, num_features, forecast_horizon, num_objects, embedding_dim=8):
    input_seq = Input(shape=(window_size, num_features), name='input_seq')
    obj_id_input = Input(shape=(1,), name='id_input')

    obj_embed = Embedding(input_dim=num_objects, output_dim=embedding_dim)(obj_id_input)
    obj_embed_vector = Reshape((embedding_dim,))(obj_embed)
    obj_context = Dense(64, activation='relu')(obj_embed_vector)

    combined_input = Concatenate()([input_seq[:, -1, :], obj_context])
    dense_out = Dense(64, activation='relu')(combined_input)
    output = Dense(forecast_horizon)(dense_out)

    model = Model(inputs=[input_seq, obj_id_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# --- Metrics ---
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    max_error = np.max(np.abs(y_true - y_pred))
    min_error = np.min(np.abs(y_true - y_pred))

    print(f'MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}')
    print(f'Max Absolute Error: {max_error:.3f}, Min Absolute Error: {min_error:.3f}')
    return mae, rmse, r2

# --- Plotting ---
def plot_predictions(y_true, y_pred, title='True vs Predicted', n=200):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:n], label='True')
    plt.plot(y_pred[:n], label='Predicted')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Load and Process Data ---
data = pd.read_csv('data/processed/with_anode.csv')
data['volume'] = (data['BATH_HEIGHT']/100) * 6.9 * 4.12
data = preprocess_data(data)

features = [col for col in data.columns if col not in ['DATE_STAMP', 'TEMPERATURE', 'OBJT_ID', 'OBJT_ID_IDX']]
target_col = 'TEMPERATURE'
window_size, forecast_horizon = 7, 1

object_ids = data['OBJT_ID'].unique()
num_objects = len(object_ids)
obj_to_idx = {obj: i for i, obj in enumerate(object_ids)}
data['OBJT_ID_IDX'] = data['OBJT_ID'].map(obj_to_idx)

X, y, ids, temperature_scalers = create_sequences(data, object_ids, features, target_col, window_size, forecast_horizon)
X_id = np.array([obj_to_idx[i] for i in ids]).reshape(-1, 1)

# --- Train/Test Split ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_id_train, X_id_test = X_id[:split], X_id[split:]
ids_train, ids_test = ids[:split], ids[split:]

# --- Build and Train Model ---
model = build_model(window_size, len(features), forecast_horizon, num_objects)
model.fit([X_train, X_id_train], y_train, validation_split=0.1, epochs=10, batch_size=32)

# --- Predict ---
y_pred = model.predict([X_test, X_id_test])

# --- Inverse Transform ---
y_true_real, y_pred_real = [], []
for i in range(len(y_test)):
    obj_id = ids_test[i]
    scaler = temperature_scalers[obj_id]
    y_true_real.append(scaler.inverse_transform(y_test[i]))
    y_pred_real.append(scaler.inverse_transform(y_pred[i]))

y_true_real = np.array(y_true_real).flatten()
y_pred_real = np.array(y_pred_real).flatten()

# --- Evaluate and Visualize ---
evaluate_model(y_true_real, y_pred_real)
plot_predictions(y_true_real, y_pred_real, title='True vs Predicted (Real Values)')
