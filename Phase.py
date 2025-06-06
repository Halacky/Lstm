import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, LayerNormalization, Dropout
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Конфигурация
CONFIG = {
    'window_size': 7,
    'forecast_horizon': 1,
    'batch_size': 64,
    'phase1_epochs': 100,
    'phase2_epochs': 150,
    'phase3_epochs': 100,
    'final_epochs': 200,
    'lstm_units': 128,
    'learning_rate': 0.001
}

# 1. Загрузка и подготовка данных
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['DATE_STAMP'] = pd.to_datetime(df['DATE_STAMP'])
    df.sort_values(['OBJT_ID', 'DATE_STAMP'], inplace=True)
    return df

# 2. Создание последовательностей с группировкой по объектам
def create_sequences_grouped(data, train_ratio=0.8):
    X_train, X_test, y_train, y_test = [], [], [], []
    feature_cols = ['TEMPERATURE', 'BATH_HEIGHT', 'VOLTAGE']  # Пример признаков
    
    # Масштабируем данные для каждого объекта отдельно
    for obj_id, group in data.groupby('OBJT_ID'):
        scaler = StandardScaler()
        obj_values = scaler.fit_transform(group[feature_cols].values)
        
        # Разделение на train/test внутри каждого объекта
        split_idx = int(len(obj_values) * train_ratio)
        
        # Генерация последовательностей для train
        for i in range(split_idx - CONFIG['window_size'] - CONFIG['forecast_horizon'] + 1):
            X_train.append(obj_values[i:i+CONFIG['window_size']])
            y_train.append(obj_values[i+CONFIG['window_size']:i+CONFIG['window_size']+CONFIG['forecast_horizon'], 0])
        
        # Генерация последовательностей для test
        for i in range(split_idx, len(obj_values) - CONFIG['window_size'] - CONFIG['forecast_horizon'] + 1):
            X_test.append(obj_values[i:i+CONFIG['window_size']])
            y_test.append(obj_values[i+CONFIG['window_size']:i+CONFIG['window_size']+CONFIG['forecast_horizon'], 0])
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# 3. Архитектура модели
def build_base_model(input_shape):
    """Базовая модель для фазы 1"""
    inputs = Input(shape=input_shape)
    x = LSTM(CONFIG['lstm_units'], return_sequences=True)(inputs)
    x = LayerNormalization()(x)
    x = LSTM(CONFIG['lstm_units']//2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(CONFIG['forecast_horizon'])(x)
    return Model(inputs, outputs)

def add_deviation_head(base_model, input_shape):
    """Добавляем head для предсказания отклонений"""
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    
    # Head для отклонений
    deviation = Dense(32, activation='relu')(x)
    deviation = Dense(1, name='deviation_output')(deviation)
    
    return Model(inputs, [x, deviation])

# 4. Фазы обучения
def train_phase1(model, X_train, y_train, X_val, y_val):
    """Фаза 1: Обучение базовой модели на всех данных"""
    model.compile(optimizer=Adam(CONFIG['learning_rate']), 
                  loss='mae')
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['phase1_epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
        verbose=1
    )
    return history

def train_phase2(model, X_train, y_train, X_val, y_val):
    """Фаза 2: Тонкая настройка на индивидуальных временных рядах"""
    # Замораживаем часть слоев
    for layer in model.layers[:-3]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(CONFIG['learning_rate']/10),
                  loss='mae')
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['phase2_epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
        verbose=1
    )
    return history

def train_phase3(model, X_train, y_train, X_val, y_val, temp_mean):
    """Фаза 3: Обучение на отклонениях от среднего"""
    # Преобразуем целевые значения в отклонения
    y_train_dev = y_train - temp_mean
    y_val_dev = y_val - temp_mean
    
    # Создаем модель с двумя выходами
    deviation_model = add_deviation_head(model, X_train.shape[1:])
    
    # Компилируем с двумя потерями
    deviation_model.compile(
        optimizer=Adam(CONFIG['learning_rate']/20),
        loss={'deviation_output': 'mae', 'dense_1': 'mae'},
        loss_weights={'deviation_output': 0.8, 'dense_1': 0.2}
    )
    
    history = deviation_model.fit(
        X_train,
        {'deviation_output': y_train_dev, 'dense_1': y_train},
        validation_data=(X_val, {'deviation_output': y_val_dev, 'dense_1': y_val}),
        epochs=CONFIG['phase3_epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
        verbose=1
    )
    return deviation_model, history

def train_final(model, X_train, y_train, X_val, y_val):
    """Финальная фаза: полная настройка"""
    # Размораживаем все слои
    for layer in model.layers:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(CONFIG['learning_rate']/50),
        loss='mae'
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['final_epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[EarlyStopping(patience=25, restore_best_weights=True)],
        verbose=1
    )
    return history

# Основной пайплайн
def main():
    # 1. Загрузка данных
    data = load_data('electrolyzers.csv')
    
    # 2. Подготовка последовательностей
    X_train, X_test, y_train, y_test = create_sequences_grouped(data)
    
    # Дополнительное разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 3. Вычисляем среднюю температуру для фазы 3
    temp_mean = np.mean(y_train)
    
    # 4. Построение модели
    base_model = build_base_model(X_train.shape[1:])
    
    # 5. Фаза 1: Базовое обучение
    print("=== Phase 1: Base Training ===")
    history1 = train_phase1(base_model, X_train, y_train, X_val, y_val)
    
    # 6. Фаза 2: Индивидуальная настройка
    print("\n=== Phase 2: Individual Tuning ===")
    history2 = train_phase2(base_model, X_train, y_train, X_val, y_val)
    
    # 7. Фаза 3: Обучение на отклонениях
    print("\n=== Phase 3: Deviation Focus ===")
    deviation_model, history3 = train_phase3(
        base_model, X_train, y_train, X_val, y_val, temp_mean
    )
    
    # 8. Финальная фаза
    print("\n=== Final Phase ===")
    final_model = clone_model(base_model)
    final_model.set_weights(base_model.get_weights())
    history_final = train_final(final_model, X_train, y_train, X_val, y_val)
    
    # 9. Оценка
    evaluate_model(final_model, X_test, y_test, temp_mean)

def evaluate_model(model, X_test, y_test, temp_mean):
    """Оценка модели с визуализацией"""
    y_pred = model.predict(X_test).flatten()
    
    # Вычисляем отклонения
    y_dev = y_test.flatten() - temp_mean
    pred_dev = y_pred - temp_mean
    
    # Метрики
    mae_total = np.mean(np.abs(y_test.flatten() - y_pred))
    mae_dev = np.mean(np.abs(y_dev - pred_dev))
    
    print(f"\nTotal MAE: {mae_total:.3f}")
    print(f"Deviation MAE: {mae_dev:.3f}")
    
    # Визуализация
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_dev, pred_dev, alpha=0.3)
    plt.plot([min(y_dev), max(y_dev)], [min(y_dev), max(y_dev)], 'r--')
    plt.xlabel('True Deviations')
    plt.ylabel('Predicted Deviations')
    plt.title('Deviation Analysis')
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()

if __name__ == '__main__':
    main()
