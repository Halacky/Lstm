import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding, Reshape, BatchNormalization, Dropout, Layer
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --- Enhanced Attention Layer ---
class EnhancedAttentionLayer(Layer):
    def __init__(self, units=64, **kwargs):
        super(EnhancedAttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W1 = self.add_weight(name='attention_weights1',
                                  shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='attention_weights2',
                                  shape=(self.units, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        super(EnhancedAttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.keras.backend.dot(x, self.W1) + self.b)
        e = tf.keras.backend.dot(e, self.W2)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# --- Preprocessing ---
def preprocess_data(data):
    data['DATE_STAMP'] = pd.to_datetime(data['DATE_STAMP'])
    data.sort_values(by=['OBJT_ID', 'DATE_STAMP'], inplace=True)
    data = data.groupby('OBJT_ID').apply(lambda group: group.fillna(method='bfill')).reset_index(drop=True)
    return data

# --- Sequence Creation ---
def create_sequences(data, obj_ids, features, target_col, window_size, forecast_horizon):
    X, y, ids = [], [], []
    temperature_scalers = {}
    data.fillna(method='bfill', inplace=True)
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

# --- Model ---
def build_fixed_model(window_size, num_features, forecast_horizon, num_objects, embedding_dim=4):
    input_seq = Input(shape=(window_size, num_features), name='input_seq')
    obj_id_input = Input(shape=(1,), name='id_input')

    lstm_out = LSTM(16, return_sequences=True)(input_seq)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.5)(lstm_out)
    attention_out = EnhancedAttentionLayer(units=16)(lstm_out)

    obj_embed = Embedding(input_dim=num_objects, output_dim=embedding_dim)(obj_id_input)
    obj_embed = Reshape((embedding_dim,))(obj_embed)
    obj_context = Dense(8, activation='relu')(obj_embed)
    obj_context = BatchNormalization()(obj_context)
    obj_context = Dropout(0.5)(obj_context)

    combined = Concatenate()([attention_out, obj_context])
    x = Dense(8, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    main_output = Dense(forecast_horizon, name='main_output')(x)

    deviation_layer = Embedding(input_dim=num_objects, output_dim=forecast_horizon, name="deviation_lookup")(obj_id_input)
    deviation_layer = Reshape((forecast_horizon,))(deviation_layer)

    final_output = tf.keras.layers.Add(name='final_output')([main_output, deviation_layer])

    model = Model(inputs=[input_seq, obj_id_input], outputs=final_output)
    model.compile(optimizer=AdamW(learning_rate=0.01), loss='mae', metrics=['mae'])
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

# --- Split sequences for local time-based train/test ---
def group_time_split_sequences(X, y, ids, test_size=0.2):
    X_train_list, y_train_list, ids_train_list = [], [], []
    X_test_list, y_test_list, ids_test_list = [], [], []

    unique_ids = np.unique(ids)
    for obj_id in unique_ids:
        idxs = np.where(ids == obj_id)[0]
        n = len(idxs)
        split_idx = int(n * (1 - test_size))

        train_idx = idxs[:split_idx]
        test_idx = idxs[split_idx:]

        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])
        ids_train_list.append(ids[train_idx])

        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])
        ids_test_list.append(ids[test_idx])

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    ids_train = np.concatenate(ids_train_list)
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)
    ids_test = np.concatenate(ids_test_list)

    return X_train, y_train, ids_train, X_test, y_test, ids_test

# --- Main ---
data = pd.read_csv('/home/golovanks/ts/uadd-forecasting/research/experiments/golovan_0_create_dataset/data/processed/result_dataset.csv')
data['volume'] = (data['BATH_HEIGHT']/100) * 6.9 * 4.12
data = preprocess_data(data)

features = [col for col in data.columns if col not in ['DATE_STAMP', 'TEMPERATURE', 'OBJT_ID', 'OBJT_ID_IDX', 'Unnamed: 0']]
target_col = 'TEMPERATURE'
window_size, forecast_horizon = 7, 1

object_ids = data['OBJT_ID'].unique()
num_objects = len(object_ids)
obj_to_idx = {obj: i for i, obj in enumerate(object_ids)}
data['OBJT_ID_IDX'] = data['OBJT_ID'].map(obj_to_idx)

X, y, ids, temperature_scalers = create_sequences(data, object_ids, features, target_col, window_size, forecast_horizon)

# === Глобальный (перемешанный) датасет ===
X_global, y_global, ids_global = shuffle(X, y, ids, random_state=42)
X_id_global = np.array([obj_to_idx[i] for i in ids_global]).reshape(-1, 1)

print(f'Global dataset size: {len(X_global)}')

# Обучение на глобальном датасете
model = build_fixed_model(window_size, len(features), forecast_horizon, num_objects)
history_global = model.fit(
    [X_global, X_id_global], y_global,
    validation_split=0.1,
    epochs=20,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# === Локальный (по времени внутри объектов) датасет ===
X_train_local, y_train_local, ids_train_local, X_test_local, y_test_local, ids_test_local = group_time_split_sequences(X, y, ids, test_size=0.2)
X_id_train_local = np.array([obj_to_idx[i] for i in ids_train_local]).reshape(-1, 1)
X_id_test_local = np.array([obj_to_idx[i] for i in ids_test_local]).reshape(-1, 1)

print(f'Local train size: {len(X_train_local)}, test size: {len(X_test_local)}')

# Дообучение модели на локальных данных
history_local = model.fit(
    [X_train_local, X_id_train_local], y_train_local,
    validation_data=([X_test_local, X_id_test_local], y_test_local),
    epochs=20,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# --- Предсказания ---
y_pred = model.predict([X_test_local, X_id_test_local])

# Обратное масштабирование
y_true_real, y_pred_real = [], []
for i in range(len(y_test_local)):
    obj_id = ids_test_local[i]
    scaler = temperature_scalers[obj_id]
    y_true_real.append(scaler.inverse_transform(y_test_local[i]))
    y_pred_real.append(scaler.inverse_transform(y_pred[i].reshape(1, -1)))

y_true_real = np.array(y_true_real).flatten()
y_pred_real = np.array(y_pred_real).flatten()

# --- Оценка ---
evaluate_model(y_true_real, y_pred_real)
plot_predictions(y_true_real, y_pred_real, title='True vs Predicted (Real Values)')

# --- Анализ по объектам ---
obj_performance = {}
for obj_id in np.unique(ids_test_local):
    mask = ids_test_local == obj_id
    obj_y_true = y_true_real[mask]
    obj_y_pred = y_pred_real[mask]
    mae = mean_absolute_error(obj_y_true, obj_y_pred)
    obj_performance[obj_id] = mae

print("\nPer-object MAE:")
for obj_id, mae in sorted(obj_performance.items(), key=lambda x: x[1]):
    print(f"Object {obj_id}: MAE = {mae:.3f}")
