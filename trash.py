import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding, Reshape, BatchNormalization, Dropout, Layer
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
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

# --- Data Preprocessing ---
def preprocess_data(data):
    data['DATE_STAMP'] = pd.to_datetime(data['DATE_STAMP'])
    data.sort_values(by=['OBJT_ID', 'DATE_STAMP'], inplace=True)
    data = data.groupby('OBJT_ID').apply(lambda group: group.fillna(method='bfill')).reset_index(drop=True)
    return data

# --- Sequence Creation ---
def create_sequences(data, obj_ids, features, target_col, window_size, forecast_horizon, shuffle_within_obj=True):
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

        # Здесь выбор варианта — перемешивать последовательности внутри объекта или идти строго по порядку
        indices = list(range(len(obj_data) - window_size - forecast_horizon))
        if shuffle_within_obj:
            np.random.shuffle(indices)

        for i in indices:
            X.append(scaled_features[i:i+window_size])
            y.append(scaled_target[i+window_size:i+window_size+forecast_horizon])
            ids.append(obj_id)

    return np.array(X), np.array(y), np.array(ids), temperature_scalers

# --- Model Builder ---
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

# --- Load and preprocess data ---
data = pd.read_csv('/home/golovanks/ts/uadd-forecasting/research/experiments/golovan_0_create_dataset/data/processed/result_dataset.csv')
data['volume'] = (data['BATH_HEIGHT'] / 100) * 6.9 * 4.12
data = preprocess_data(data)

features = [col for col in data.columns if col not in ['DATE_STAMP', 'TEMPERATURE', 'OBJT_ID', 'OBJT_ID_IDX', 'Unnamed: 0']]
target_col = 'TEMPERATURE'
window_size, forecast_horizon = 7, 1

object_ids = data['OBJT_ID'].unique()
num_objects = len(object_ids)
obj_to_idx = {obj: i for i, obj in enumerate(object_ids)}
data['OBJT_ID_IDX'] = data['OBJT_ID'].map(obj_to_idx)

# --- Создаем глобальные данные: перемешка по времени внутри каждого OBJT_ID ---
X_global, y_global, ids_global, scalers_global = create_sequences(data, object_ids, features, target_col, window_size, forecast_horizon, shuffle_within_obj=True)
X_id_global = np.array([obj_to_idx[i] for i in ids_global]).reshape(-1, 1)

# --- Создаем локальные данные: строго по времени, не перемешиваем, и выделяем train/test по OBJT_ID ---
# Делим OBJT_ID на train/test (например 80/20)
train_obj_ids = object_ids[:int(0.8 * num_objects)]
test_obj_ids = object_ids[int(0.8 * num_objects):]

X_local_train, y_local_train, ids_local_train, scalers_local_train = create_sequences(data, train_obj_ids, features, target_col, window_size, forecast_horizon, shuffle_within_obj=False)
X_id_local_train = np.array([obj_to_idx[i] for i in ids_local_train]).reshape(-1, 1)

X_local_test, y_local_test, ids_local_test, scalers_local_test = create_sequences(data, test_obj_ids, features, target_col, window_size, forecast_horizon, shuffle_within_obj=False)
X_id_local_test = np.array([obj_to_idx[i] for i in ids_local_test]).reshape(-1, 1)

# --- Создаем tf.data.Dataset ---
batch_size = 64

dataset_global = tf.data.Dataset.from_tensor_slices((X_global, X_id_global, y_global))
dataset_global = dataset_global.shuffle(10000, seed=42).batch(batch_size).repeat()

dataset_local = tf.data.Dataset.from_tensor_slices((X_local_train, X_id_local_train, y_local_train))
dataset_local = dataset_local.batch(batch_size).repeat()

train_dataset = tf.data.Dataset.zip((dataset_global, dataset_local))

# --- Построение модели ---
model = build_fixed_model(window_size, len(features), forecast_horizon, num_objects)
optimizer = AdamW(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanAbsoluteError()

# --- Метрики для отслеживания ---
train_loss_global = tf.keras.metrics.Mean(name='train_loss_global')
train_loss_local = tf.keras.metrics.Mean(name='train_loss_local')
train_loss_total = tf.keras.metrics.Mean(name='train_loss_total')

@tf.function
def train_step(batch_global, batch_local):
    x_global, id_global, y_global = batch_global
    x_local, id_local, y_local = batch_local
    
    with tf.GradientTape() as tape:
        pred_global = model([x_global, id_global], training=True)
        pred_local = model([x_local, id_local], training=True)
        
        loss_global = loss_fn(y_global, pred_global)
        loss_local = loss_fn(y_local, pred_local)
        
        total_loss = loss_global + loss_local

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_global(loss_global)
    train_loss_local(loss_local)
    train_loss_total(total_loss)

EPOCHS = 50
steps_per_epoch = 100  # можно менять под размер данных

for epoch in range(EPOCHS):
    train_loss_global.reset_states()
    train_loss_local.reset_states()
    train_loss_total.reset_states()

    for step, (batch_global, batch_local) in enumerate(train_dataset.take(steps_per_epoch)):
        train_step(batch_global, batch_local)

    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Global Loss: {train_loss_global.result():.4f}, "
          f"Local Loss: {train_loss_local.result():.4f}, "
          f"Total Loss: {train_loss_total.result():.4f}")

# --- Тестирование на локальных данных из тестового сета ---
y_pred = model.predict([X_local_test, X_id_local_test])

# Обратное преобразование к реальным значениям
y_true_real, y_pred_real = [], []
for i in range(len(y_local_test)):
    obj_id = ids_local_test[i]
    scaler = scalers_local_test[obj_id]
    y_true_real.append(scaler.inverse_transform(y_local_test[i]))
    y_pred_real.append(scaler.inverse_transform(y_pred[i].reshape(1, -1)))

y_true_real = np.array(y_true_real).flatten()
y_pred_real = np.array(y_pred_real).flatten()

# --- Оценка и визуализация ---
evaluate_model(y_true_real, y_pred_real)
plot_predictions(y_true_real, y_pred_real, title='True vs Predicted (Local Test Set)')
