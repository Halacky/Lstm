import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Concatenate, Embedding, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Конфигурация
CONFIG = {
    'daily_features': ['TEMPERATURE', 'CONDUCTIVITY'],
    'minute_features': ['VOLTAGE', 'CURRENT', 'ALUMINA'],
    'window_size': 30,  # дней для дневных данных
    'minute_seq_len': 24*7,  # часов (1 неделя)
    'batch_size': 32,
    'epochs': 200,
    'embedding_dim': 16,
    'lstm_units': 128,
    'test_size': 0.15,
    'val_size': 0.15
}

# 1. Загрузка и подготовка данных
def load_data(daily_path, minute_path):
    daily_df = pd.read_csv(daily_path, parse_dates=['DATE'])
    minute_df = pd.read_csv(minute_path, parse_dates=['TIMESTAMP'])
    return daily_df, minute_df

# 2. Предобработка дневных данных
def prepare_daily(daily_df):
    features = []
    targets = []
    obj_ids = []
    
    for obj_id, group in daily_df.groupby('OBJT_ID'):
        # Скользящие статистики
        group['TEMP_7D_MA'] = group['TEMPERATURE'].rolling(7).mean()
        group['TEMP_30D_STD'] = group['TEMPERATURE'].rolling(30).std()
        
        # Временные фичи
        group['DAY_OF_WEEK'] = group['DATE'].dt.dayofweek
        group['MONTH'] = group['DATE'].dt.month
        
        # Удаление NaN после rolling
        group = group.dropna()
        
        # Нормализация
        scaler = RobustScaler()
        scaled = scaler.fit_transform(group[CONFIG['daily_features'] + ['TEMP_7D_MA', 'TEMP_30D_STD']])
        
        # Формирование последовательностей
        for i in range(len(group) - CONFIG['window_size'] - 1):
            features.append(scaled[i:i+CONFIG['window_size']])
            targets.append(scaled[i+CONFIG['window_size'], 0])  # Температура
            obj_ids.append(obj_id)
    
    return np.array(features), np.array(targets), np.array(obj_ids)

# 3. Предобработка 3-минутных данных
def prepare_minute(minute_df):
    features = []
    obj_ids = []
    
    for obj_id, group in minute_df.groupby('OBJT_ID'):
        # Агрегация по часам
        resampled = group.resample('1H', on='TIMESTAMP').agg({
            'VOLTAGE': ['mean', 'std'],
            'CURRENT': ['mean', 'max'],
            'ALUMINA': 'sum'
        })
        resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
        
        # Нормализация
        scaler = StandardScaler()
        scaled = scaler.fit_transform(resampled)
        
        # Разбиение на последовательности
        for i in range(len(resampled) - CONFIG['minute_seq_len']):
            features.append(scaled[i:i+CONFIG['minute_seq_len']])
            obj_ids.append(obj_id)
    
    return np.array(features), np.array(obj_ids)

# 4. Синхронизация данных
def align_data(daily_X, daily_y, daily_ids, minute_X, minute_ids):
    aligned_X = []
    aligned_minute = []
    aligned_y = []
    
    unique_ids = np.unique(daily_ids)
    for obj_id in unique_ids:
        daily_idx = np.where(daily_ids == obj_id)[0]
        minute_idx = np.where(minute_ids == obj_id)[0]
        
        min_len = min(len(daily_idx), len(minute_idx))
        if min_len == 0:
            continue
            
        aligned_X.append(daily_X[daily_idx[:min_len]])
        aligned_minute.append(minute_X[minute_idx[:min_len]])
        aligned_y.append(daily_y[daily_idx[:min_len]])
    
    return np.concatenate(aligned_X), np.concatenate(aligned_minute), np.concatenate(aligned_y)

# 5. Построение модели
def build_model(num_objects):
    # Входы
    daily_input = Input(shape=(CONFIG['window_size'], len(CONFIG['daily_features']) + 2), name='daily_input')
    minute_input = Input(shape=(CONFIG['minute_seq_len'], len(CONFIG['minute_features'])*2), name='minute_input')
    obj_input = Input(shape=(1,), name='obj_id', dtype='int32')
    
    # Эмбеддинг объектов
    obj_embed = Embedding(num_objects, CONFIG['embedding_dim'])(obj_input)
    obj_embed = Dense(8, activation='swish')(Flatten()(obj_embed))
    
    # Обработка дневных данных
    x_daily = LSTM(CONFIG['lstm_units'], return_sequences=True)(daily_input)
    x_daily = LayerNormalization()(x_daily)
    daily_pool = GlobalAveragePooling1D()(x_daily)
    
    # Обработка минутных данных
    x_minute = GRU(64, return_sequences=True)(minute_input)
    x_minute = LayerNormalization()(x_minute)
    
    # Механизм внимания
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x_minute, x_minute)
    minute_pool = GlobalAveragePooling1D()(attention)
    
    # Объединение
    combined = Concatenate()([daily_pool, minute_pool, obj_embed])
    x = Dense(64, activation='swish')(combined)
    x = Dropout(0.3)(x)
    
    # Выходы
    main_output = Dense(1, name='temperature')(x)
    deviation = Dense(1, activation='tanh', name='deviation')(x)
    
    return Model(inputs=[daily_input, minute_input, obj_input], outputs=[main_output, deviation])

# 6. Обучение
def train_model(model, X_train, X_minute_train, obj_train, y_train, X_val, X_minute_val, obj_val, y_val):
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'temperature': 'mae', 'deviation': 'mse'},
        loss_weights={'temperature': 0.6, 'deviation': 0.4},
        metrics={'temperature': ['mae', 'mse']}
    )
    
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    history = model.fit(
        {'daily_input': X_train, 'minute_input': X_minute_train, 'obj_id': obj_train},
        {'temperature': y_train, 'deviation': y_train - np.mean(y_train)},
        validation_data=(
            {'daily_input': X_val, 'minute_input': X_minute_val, 'obj_id': obj_val},
            {'temperature': y_val, 'deviation': y_val - np.mean(y_val)}
        ),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    return history

# 7. Оценка
def evaluate_model(model, X_test, X_minute_test, obj_test, y_test, scaler):
    # Предсказание
    y_pred, _ = model.predict({'daily_input': X_test, 'minute_input': X_minute_test, 'obj_id': obj_test})
    y_pred = y_pred.flatten()
    
    # Обратное масштабирование
    y_test_real = scaler.inverse_transform(np.c_[y_test, np.zeros_like(y_test)])[:, 0]
    y_pred_real = scaler.inverse_transform(np.c_[y_pred, np.zeros_like(y_pred)])[:, 0]
    
    # Метрики
    metrics = {
        'MAE': mean_absolute_error(y_test_real, y_pred_real),
        'R2': r2_score(y_test_real, y_pred_real),
        'MinAE': np.min(np.abs(y_test_real - y_pred_real)),
        'MaxAE': np.max(np.abs(y_test_real - y_pred_real))
    }
    
    # Графики
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_real[:200], label='True')
    plt.plot(y_pred_real[:200], label='Predicted')
    plt.title(f"Temperature Prediction\nMAE: {metrics['MAE']:.2f}, R2: {metrics['R2']:.2f}")
    plt.legend()
    plt.savefig('predictions.png')
    plt.close()
    
    return metrics

# Основной пайплайн
def main():
    # 1. Загрузка
    daily_df, minute_df = load_data('daily.csv', 'minute.csv')
    
    # 2. Подготовка
    daily_X, daily_y, daily_ids = prepare_daily(daily_df)
    minute_X, minute_ids = prepare_minute(minute_df)
    
    # 3. Синхронизация
    X, X_minute, y = align_data(daily_X, daily_y, daily_ids, minute_X, minute_ids)
    obj_ids = np.unique(daily_ids, return_inverse=True)[1]
    
    # 4. Разделение
    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG['test_size'])
    train_idx, test_idx = next(gss.split(X, groups=obj_ids))
    
    X_train, X_test = X[train_idx], X[test_idx]
    Xm_train, Xm_test = X_minute[train_idx], X_minute[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    obj_train, obj_test = obj_ids[train_idx], obj_ids[test_idx]
    
    # Доп. разделение на train/val
    X_train, X_val, Xm_train, Xm_val, y_train, y_val, obj_train, obj_val = train_test_split(
        X_train, Xm_train, y_train, obj_train, test_size=CONFIG['val_size'], random_state=42
    )
    
    # 5. Построение и обучение модели
    model = build_model(len(np.unique(obj_ids)))
    history = train_model(model, X_train, Xm_train, obj_train, y_train, X_val, Xm_val, obj_val, y_val)
    
    # 6. Оценка
    metrics = evaluate_model(model, X_test, Xm_test, obj_test, y_test, scaler)
    print("\nMetrics on Test Set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()


def train_test_split_timeseries(data, test_size=0.2):
    """Корректное разделение временных рядов по объектам"""
    train_data, test_data = [], []
    
    for obj_id, group in data.groupby('OBJT_ID'):
        group = group.sort_values('DATE')  # Сортировка по времени
        
        # Вычисляем индекс для разделения
        split_idx = int(len(group) * (1 - test_size))
        
        # Разделяем данные
        train_data.append(group.iloc[:split_idx])
        test_data.append(group.iloc[split_idx:])
    
    return pd.concat(train_data), pd.concat(test_data)

def prepare_datasets(daily_df, minute_df):
    """Подготовка датасетов с правильным разделением"""
    # 1. Разделяем дневные данные
    daily_train, daily_test = train_test_split_timeseries(daily_df)
    
    # 2. Для минутных данных используем те же временные отрезки
    minute_train = minute_df[minute_df['TIMESTAMP'] < daily_train['DATE'].max()]
    minute_test = minute_df[minute_df['TIMESTAMP'] >= daily_train['DATE'].max()]
    
    # 3. Подготовка фичей с сохранением масштабировщиков
    X_train, Xm_train, y_train, scaler = prepare_features(daily_train, minute_train)
    X_test, Xm_test, y_test, _ = prepare_features(daily_test, minute_test, scaler)
    
    return (X_train, Xm_train, y_train), (X_test, Xm_test, y_test), scaler

def prepare_features(daily_df, minute_df, scaler=None):
    """Подготовка фичей с учетом временного порядка"""
    # Дневные данные
    daily_features = []
    targets = []
    obj_ids = []
    
    temp_scaler = scaler or RobustScaler()
    
    for obj_id, group in daily_df.groupby('OBJT_ID'):
        group = group.sort_values('DATE')
        
        # Масштабирование
        if scaler is None:
            temp_values = temp_scaler.fit_transform(group[['TEMPERATURE']])
        else:
            temp_values = temp_scaler.transform(group[['TEMPERATURE']])
        
        # Формирование последовательностей
        for i in range(len(group) - CONFIG['window_size'] - 1):
            daily_features.append(group.iloc[i:i+CONFIG['window_size']][CONFIG['daily_features']].values)
            targets.append(temp_values[i+CONFIG['window_size']])
            obj_ids.append(obj_id)
    
    # Минутные данные (агрегация по часам)
    minute_features = []
    minute_obj_ids = []
    
    for obj_id, group in minute_df.groupby('OBJT_ID'):
        group = group.sort_values('TIMESTAMP')
        resampled = group.resample('1H', on='TIMESTAMP').mean()
        
        # Сопоставление с дневными данными
        for i in range(len(daily_features)):
            if obj_ids[i] == obj_id:
                hour_start = daily_df.iloc[i]['DATE'] - pd.Timedelta(hours=CONFIG['minute_seq_len'])
                hour_data = resampled.loc[hour_start:daily_df.iloc[i]['DATE']].values
                if len(hour_data) == CONFIG['minute_seq_len']:
                    minute_features.append(hour_data)
                    minute_obj_ids.append(obj_id)
    
    return (np.array(daily_features), 
            np.array(minute_features), 
            np.array(targets), 
            temp_scaler)

# Модифицированный основной пайплайн
def main():
    # 1. Загрузка данных
    daily_df, minute_df = load_data('daily.csv', 'minute.csv')
    
    # 2. Корректное разделение на train/test
    (X_train, Xm_train, y_train), 
     (X_test, Xm_test, y_test), scaler = prepare_datasets(daily_df, minute_df)
    
    # 3. Дополнительное разделение на train/val (тоже с учетом временного порядка)
    X_train, X_val, Xm_train, Xm_val, y_train, y_val = time_based_split(
        X_train, Xm_train, y_train, val_size=0.15
    )
    
    # 4. Построение и обучение модели
    model = build_model(len(np.unique(np.concatenate([X_train[:,0], X_test[:,0]]))))
    history = train_model(model, X_train, Xm_train, y_train, X_val, Xm_val, y_val)
    
    # 5. Оценка
    metrics = evaluate_model(model, X_test, Xm_test, y_test, scaler)

def time_based_split(X, Xm, y, val_size):
    """Разделение с сохранением временного порядка"""
    split_idx = int(len(X) * (1 - val_size))
    return (X[:split_idx], X[split_idx:], 
            Xm[:split_idx], Xm[split_idx:],
            y[:split_idx], y[split_idx:])


plt.figure(figsize=(12, 6))
for obj_id in daily_df['OBJT_ID'].unique()[:5]:  # Первые 5 объектов
    obj_data = daily_df[daily_df['OBJT_ID'] == obj_id]
    plt.plot(obj_data['DATE'], obj_data['TEMPERATURE'], 
             label=f'Obj {obj_id}', alpha=0.7)
plt.axvline(x=daily_train['DATE'].max(), color='r', linestyle='--', 
            label='Train/Test Split')
plt.legend()
plt.title('Temporal Split Visualization')
plt.savefig('time_split.png')
