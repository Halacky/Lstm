import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics, optimizers, backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ==================== Улучшенные слои модели ====================
class MultiHeadTemporalAttention(layers.Layer):
    def __init__(self, n_heads=4, key_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.key_dim = key_dim
        
    def build(self, input_shape):
        self.query = layers.Dense(self.key_dim * self.n_heads)
        self.key = layers.Dense(self.key_dim * self.n_heads)
        self.value = layers.Dense(self.key_dim * self.n_heads)
        self.combine = layers.Dense(input_shape[-1])
        
    def call(self, x):
        # Reshape for multi-head attention
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Split into multiple heads
        q = tf.reshape(q, (-1, tf.shape(q)[1], self.n_heads, self.key_dim))
        k = tf.reshape(k, (-1, tf.shape(k)[1], self.n_heads, self.key_dim))
        v = tf.reshape(v, (-1, tf.shape(v)[1], self.n_heads, self.key_dim))
        
        # Scaled dot-product attention
        attn_scores = tf.einsum('bqhd,bkhd->bhqk', q, k) / tf.sqrt(tf.cast(self.key_dim, tf.float32))
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        output = tf.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        # Combine heads
        output = tf.reshape(output, (-1, tf.shape(output)[1], self.n_heads * self.key_dim))
        return self.combine(output)

class ObjectSpecificAdaptation(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.gamma_net = models.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(input_shape[-1], activation='sigmoid')
        ])
        self.beta_net = models.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(input_shape[-1])
        ])
        
    def call(self, inputs):
        x, obj_embedding = inputs
        gamma = self.gamma_net(obj_embedding)
        beta = self.beta_net(obj_embedding)
        return x * gamma + beta

# ==================== Обработка данных ====================
class TimeSeriesPreprocessor:
    def __init__(self, window_size=14, forecast_horizon=1, test_size=0.2):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.feature_scalers = {}
        self.target_scalers = {}
        
    def preprocess(self, data, obj_col='OBJT_ID', time_col='DATE_STAMP', target_col='TEMPERATURE'):
        data[time_col] = pd.to_datetime(data[time_col])
        data.sort_values(by=[obj_col, time_col], inplace=True)
        
        # Заполнение пропусков
        data = data.groupby(obj_col).apply(
            lambda x: x.interpolate(method='linear').fillna(method='bfill')
        ).reset_index(drop=True)
        
        return data
    
    def create_sequences(self, data, features, target_col, obj_ids=None, shuffle=False):
        if obj_ids is None:
            obj_ids = data['OBJT_ID'].unique()
            
        X, y, ids, times = [], [], [], []
        
        for obj_id in tqdm(obj_ids, desc="Creating sequences"):
            obj_data = data[data['OBJT_ID'] == obj_id]
            if len(obj_data) < self.window_size + self.forecast_horizon:
                continue
                
            # Масштабирование
            if obj_id not in self.feature_scalers:
                self.feature_scalers[obj_id] = StandardScaler()
                self.target_scalers[obj_id] = StandardScaler()
                
            X_scaled = self.feature_scalers[obj_id].fit_transform(obj_data[features])
            y_scaled = self.target_scalers[obj_id].fit_transform(obj_data[[target_col]])
            
            # Создание последовательностей
            for i in range(len(obj_data) - self.window_size - self.forecast_horizon + 1):
                X.append(X_scaled[i:i+self.window_size])
                y.append(y_scaled[i+self.window_size:i+self.window_size+self.forecast_horizon])
                ids.append(obj_id)
                times.append(obj_data.iloc[i+self.window_size][time_col])
                
        X = np.array(X)
        y = np.array(y).squeeze()
        
        if shuffle:
            idx = np.random.permutation(len(X))
            X, y, ids, times = X[idx], y[idx], np.array(ids)[idx], np.array(times)[idx]
            
        return X, y, ids, times
    
    def train_test_split(self, X, y, ids, times, strategy='time'):
        if strategy == 'time':
            # Разделение по времени (последние 20% точек для каждого объекта)
            train_mask = []
            for obj_id in np.unique(ids):
                obj_mask = (ids == obj_id)
                obj_times = times[obj_mask]
                cutoff = np.quantile(obj_times, 1 - self.test_size)
                train_mask.extend((ids == obj_id) & (times < cutoff))
            train_mask = np.array(train_mask)
            
            return X[train_mask], y[train_mask], ids[train_mask], X[~train_mask], y[~train_mask], ids[~train_mask]
        
        elif strategy == 'object':
            # Разделение по объектам
            unique_ids = np.unique(ids)
            test_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * self.test_size), replace=False)
            test_mask = np.isin(ids, test_ids)
            
            return X[~test_mask], y[~test_mask], ids[~test_mask], X[test_mask], y[test_mask], ids[test_mask]

# ==================== Построение модели ====================
def build_hybrid_model(window_size, num_features, forecast_horizon, num_objects, 
                      embedding_dim=8, lstm_units=32, attention_heads=4):
    # Входы модели
    input_seq = layers.Input(shape=(window_size, num_features), name='input_seq')
    obj_id_input = layers.Input(shape=(1,), name='id_input')
    
    # Ветвь для временных зависимостей
    x = layers.LSTM(lstm_units, return_sequences=True)(input_seq)
    x = layers.BatchNormalization()(x)
    x = MultiHeadTemporalAttention(n_heads=attention_heads, key_dim=lstm_units//attention_heads)(x)
    x = layers.Dropout(0.3)(x)
    
    # Ветвь для объекто-специфичных признаков
    obj_embed = layers.Embedding(input_dim=num_objects, output_dim=embedding_dim)(obj_id_input)
    obj_embed = layers.Reshape((embedding_dim,))(obj_embed)
    
    # Комбинирование признаков
    combined = layers.Concatenate()([x, obj_embed])
    combined = layers.Dense(32, activation='relu')(combined)
    combined = layers.BatchNormalization()(combined)
    
    # Адаптация под конкретный объект
    output = ObjectSpecificAdaptation()([combined, obj_embed])
    
    # Прогноз с квантилями для оценки неопределенности
    quantiles = [0.1, 0.5, 0.9]
    outputs = []
    for q in quantiles:
        q_output = layers.Dense(forecast_horizon, name=f'q_{int(q*100)}')(output)
        outputs.append(q_output)
    
    model = models.Model(inputs=[input_seq, obj_id_input], outputs=outputs)
    return model

# ==================== Обучение и оценка ====================
class QuantileLoss:
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        
    def __call__(self, y_true, y_pred):
        losses = []
        for i, q in enumerate(self.quantiles):
            error = y_true - y_pred[i]
            loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            losses.append(loss)
        return tf.reduce_sum(losses)

def evaluate_predictions(y_true, y_pred, scalers=None, ids=None):
    if scalers is not None and ids is not None:
        # Обратное преобразование масштабирования
        y_true_rescaled = []
        y_pred_rescaled = []
        for i, obj_id in enumerate(ids):
            scaler = scalers[obj_id]
            y_true_rescaled.append(scaler.inverse_transform(y_true[i].reshape(-1, 1)))
            y_pred_rescaled.append(scaler.inverse_transform(y_pred[i].reshape(-1, 1)))
        y_true = np.concatenate(y_true_rescaled)
        y_pred = np.concatenate(y_pred_rescaled)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.grid(True)
    plt.show()
    
    return mae, rmse, r2

# ==================== Основной пайплайн ====================
def main():
    # Загрузка данных
    data = pd.read_csv('path_to_your_data.csv')
    data['volume'] = (data['BATH_HEIGHT'] / 100) * 6.9 * 4.12
    
    # Предобработка
    preprocessor = TimeSeriesPreprocessor(window_size=14, forecast_horizon=1, test_size=0.2)
    data = preprocessor.preprocess(data)
    
    # Выбор признаков
    features = [col for col in data.columns if col not in ['DATE_STAMP', 'TEMPERATURE', 'OBJT_ID', 'Unnamed: 0']]
    target_col = 'TEMPERATURE'
    
    # Создание последовательностей
    X, y, ids, times = preprocessor.create_sequences(data, features, target_col)
    
    # Разделение на train/test
    X_train, y_train, ids_train, X_test, y_test, ids_test = preprocessor.train_test_split(
        X, y, ids, times, strategy='time'
    )
    
    # Преобразование ID в индексы
    obj_to_idx = {obj: i for i, obj in enumerate(np.unique(ids))}
    ids_train_idx = np.array([obj_to_idx[i] for i in ids_train]).reshape(-1, 1)
    ids_test_idx = np.array([obj_to_idx[i] for i in ids_test]).reshape(-1, 1)
    
    # Построение модели
    model = build_hybrid_model(
        window_size=14,
        num_features=len(features),
        forecast_horizon=1,
        num_objects=len(obj_to_idx),
        lstm_units=64,
        attention_heads=4
    )
    
    # Компиляция модели
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss=QuantileLoss(),
        loss_weights=[0.2, 0.6, 0.2]  # Больший вес для медианы
    )
    
    # Обучение
    history = model.fit(
        [X_train, ids_train_idx],
        [y_train, y_train, y_train],  # Три цели для квантилей
        validation_data=([X_test, ids_test_idx], [y_test, y_test, y_test]),
        epochs=100,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Оценка
    y_pred = model.predict([X_test, ids_test_idx])[1]  # Берем медиану (q=0.5)
    evaluate_predictions(y_test, y_pred, preprocessor.target_scalers, ids_test)
    
    # Визуализация обучения
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
