import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding, RepeatVector, Layer, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# Custom Attention Layer
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


# Data Preprocessing
def preprocess_data(data):
    data['DATE_STAMP'] = pd.to_datetime(data['DATE_STAMP'])
    data.sort_values(by=['OBJT_ID', 'DATE_STAMP'], inplace=True)
    data = data.groupby('OBJT_ID').apply(lambda group: group.fillna(method='bfill')).reset_index(drop=True)
    data = data.dropna()
    return data


# Sequence generation
def create_sequences(data, obj_ids, features, target_col, window_size, forecast_horizon):
    X, y, ids = [], [], []
    target_scaler = StandardScaler()
    data[target_col] = target_scaler.fit_transform(data[[target_col]])

    feature_scaler = StandardScaler()
    data[features] = feature_scaler.fit_transform(data[features])

    for obj_id in obj_ids:
        obj_data = data[data['OBJT_ID'] == obj_id]
        for i in range(len(obj_data) - window_size - forecast_horizon):
            X.append(obj_data[features].iloc[i:i + window_size].values)
            y.append(obj_data[target_col].iloc[i + window_size:i + window_size + forecast_horizon].values)
            ids.append(obj_id)
    return np.array(X), np.array(y), np.array(ids), target_scaler


# Model
def build_model(window_size, num_features, forecast_horizon, num_objects, embedding_dim=8):
    input_seq = Input(shape=(window_size, num_features), name='input_seq')
    obj_id_input = Input(shape=(1,), name='id_input')

    obj_embed = Embedding(input_dim=num_objects, output_dim=embedding_dim)(obj_id_input)
    reshaped_embedding = Reshape((embedding_dim,))(obj_embed)
    obj_embed_repeated = RepeatVector(window_size)(reshaped_embedding)

    combined_input = Concatenate()([input_seq, obj_embed_repeated])

    common_lstm = LSTM(64, return_sequences=True)(combined_input)
    individual_lstm = LSTM(64, return_sequences=True)(combined_input)

    combined_lstm = Concatenate()([common_lstm, individual_lstm])
    attention_output = AttentionLayer()(combined_lstm)

    output = Dense(forecast_horizon)(attention_output)

    model = Model(inputs=[input_seq, obj_id_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# Evaluation
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}')
    return mae, rmse, r2


# Visualization
def plot_predictions(y_true, y_pred, title='True vs Predicted'):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:100], label='True')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Load data
data = pd.read_csv('/home/golovanks/ts/uadd-forecasting/research/experiments/golovan_0_create_dataset/data/processed/with_anode.csv')
data['volume'] = (data['BATH_HEIGHT'] / 100) * 6.9 * 4.12
data = preprocess_data(data)

features = [col for col in data.columns if col not in ['DATE_STAMP', 'TEMPERATURE', 'OBJT_ID', 'OBJT_ID_IDX']]
target_col = 'TEMPERATURE'
window_size, forecast_horizon = 7, 1

object_ids = data['OBJT_ID'].unique()
num_objects = len(object_ids)
obj_to_idx = {obj: i for i, obj in enumerate(object_ids)}
data['OBJT_ID_IDX'] = data['OBJT_ID'].map(obj_to_idx)

# Create sequences
X, y, ids, target_scaler = create_sequences(data, object_ids, features, target_col, window_size, forecast_horizon)
X_id = np.array([obj_to_idx[i] for i in ids]).reshape(-1, 1)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_id_train, X_id_test = X_id[:split], X_id[split:]

# Train
model = build_model(window_size, len(features), forecast_horizon, num_objects)
model.fit([X_train, X_id_train], y_train, validation_split=0.1, epochs=10, batch_size=32)

# Predict
y_pred = model.predict([X_test, X_id_test])
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

evaluate_model(y_test_rescaled, y_pred_rescaled)
plot_predictions(y_test_rescaled, y_pred_rescaled)
