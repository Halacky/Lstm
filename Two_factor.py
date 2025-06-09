import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Конфигурация
CONFIG = {
    'daily_features': ['TEMPERATURE', 'FACTOR2'],
    'minute_features': ['TFEED_SET', 'TFEED', 'TFLUOR', 'UTRG', 'USET', 'UADD', 
                       'DUFLTR', 'U', 'FEED_SUM_DUMPS', 'FLUOR_SUM_DUMPS', 'NOISE'],
    'window_size': 14,
    'minute_seq_len': 288,  # Последние 24 часа (5*288=1440 минут)
    'batch_size': 128,
    'epochs_base': 100,
    'epochs_fine_tune': 50,
    'lstm_units': 64,
    'learning_rate_base': 0.001,
    'learning_rate_fine_tune': 0.0001
}

# Загрузка и подготовка данных
def load_and_prepare_data(daily_path, minute_path):
    # Загрузка
    daily_df = pd.read_csv(daily_path, parse_dates=['DATE_STAMP'])
    minute_df = pd.read_csv(minute_path, parse_dates=['DATE_STAMP'])
    
    # Фильтрация общих электролизеров
    common_pots = set(daily_df['POT_ID']).intersection(set(minute_df['POT_ID']))
    daily_df = daily_df[daily_df['POT_ID'].isin(common_pots)]
    minute_df = minute_df[minute_df['POT_ID'].isin(common_pots)]
    
    # Создание последовательностей
    X_daily, Xm, y = [], [], []
    
    for pot_id in common_pots:
        pot_daily = daily_df[daily_df['POT_ID'] == pot_id].sort_values('DATE_STAMP')
        pot_minute = minute_df[minute_df['POT_ID'] == pot_id].sort_values('DATE_STAMP')
        
        for i in range(CONFIG['window_size'], len(pot_daily)):
            # Суточные данные
            X_daily.append(pot_daily[CONFIG['daily_features']].iloc[i-CONFIG['window_size']:i].values)
            y.append(pot_daily['TEMPERATURE'].iloc[i])
            
            # Минутные данные (если есть)
            target_date = pot_daily['DATE_STAMP'].iloc[i]
            minute_data = pot_minute[pot_minute['DATE_STAMP'] < target_date].iloc[-CONFIG['minute_seq_len']:]
            
            if len(minute_data) == CONFIG['minute_seq_len']:
                Xm.append(minute_data[CONFIG['minute_features']].values)
            else:
                Xm.append(np.zeros((CONFIG['minute_seq_len'], len(CONFIG['minute_features']))))
    
    X_daily, Xm, y = np.array(X_daily), np.array(Xm), np.array(y)
    
    # Разделение на train/test (учитываем что минутные данные есть только для последнего периода)
    split_idx = int(len(X_daily) * 0.8)
    
    # Полные данные (только суточные)
    X_daily_train_full = X_daily[:split_idx]
    y_train_full = y[:split_idx]
    
    X_daily_test_full = X_daily[split_idx:]
    y_test_full = y[split_idx:]
    
    # Данные с минутными признаками (последние 3.5 месяца)
    short_data_idx = int(len(X_daily) * 0.9)  # Предполагаем что минутные данные в последних 10%
    X_daily_short = X_daily[short_data_idx:]
    Xm_short = Xm[short_data_idx:]
    y_short = y[short_data_idx:]
    
    # Масштабирование
    daily_scaler = StandardScaler()
    X_daily_train_full = daily_scaler.fit_transform(X_daily_train_full.reshape(-1, len(CONFIG['daily_features']))).reshape(-1, CONFIG['window_size'], len(CONFIG['daily_features']))
    X_daily_test_full = daily_scaler.transform(X_daily_test_full.reshape(-1, len(CONFIG['daily_features']))).reshape(-1, CONFIG['window_size'], len(CONFIG['daily_features']))
    X_daily_short = daily_scaler.transform(X_daily_short.reshape(-1, len(CONFIG['daily_features']))).reshape(-1, CONFIG['window_size'], len(CONFIG['daily_features']))
    
    minute_scaler = StandardScaler()
    Xm_short = minute_scaler.fit_transform(Xm_short.reshape(-1, len(CONFIG['minute_features']))).reshape(-1, CONFIG['minute_seq_len'], len(CONFIG['minute_features']))
    
    y_scaler = StandardScaler()
    y_train_full = y_scaler.fit_transform(y_train_full.reshape(-1, 1)).flatten()
    y_test_full = y_scaler.transform(y_test_full.reshape(-1, 1)).flatten()
    y_short = y_scaler.transform(y_short.reshape(-1, 1)).flatten()
    
    return (X_daily_train_full, y_train_full), (X_daily_test_full, y_test_full), (X_daily_short, Xm_short, y_short), y_scaler

# Базовая модель (только суточные данные)
def build_base_model():
    input_layer = Input(shape=(CONFIG['window_size'], len(CONFIG['daily_features'])))
    x = LSTM(CONFIG['lstm_units'], return_sequences=True)(input_layer)
    x = LSTM(CONFIG['lstm_units'])(x)
    x = Dropout(0.3)(x)
    output = Dense(1)(x)
    
    model = Model(input_layer, output)
    model.compile(optimizer=Adam(CONFIG['learning_rate_base']), loss='mae')
    return model

# Полная модель (суточные + минутные)
def build_full_model(base_model):
    # Замораживаем базовую модель
    for layer in base_model.layers:
        layer.trainable = False
    
    # Входы
    daily_input = base_model.input
    minute_input = Input(shape=(CONFIG['minute_seq_len'], len(CONFIG['minute_features'])))
    
    # Обработка минутных данных
    xm = LSTM(CONFIG['lstm_units'])(minute_input)
    xm = Dropout(0.3)(xm)
    
    # Комбинирование
    combined = Concatenate()([base_model.output, xm])
    x = Dense(64, activation='relu')(combined)
    output = Dense(1)(x)
    
    model = Model([daily_input, minute_input], output)
    model.compile(optimizer=Adam(CONFIG['learning_rate_fine_tune']), loss='mae')
    return model

# Обучение и оценка
def main():
    # Загрузка данных
    (X_train_full, y_train_full), (X_test_full, y_test_full), (X_short, Xm_short, y_short), y_scaler = load_and_prepare_data(
        'dayly.csv', 
        '3min.csv'
    )
    
    # 1. Обучение базовой модели
    print("Training base model...")
    base_model = build_base_model()
    base_history = base_model.fit(
        X_train_full, y_train_full,
        validation_data=(X_test_full, y_test_full),
        epochs=CONFIG['epochs_base'],
        batch_size=CONFIG['batch_size'],
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # 2. Дообучение полной модели
    print("\nFine-tuning full model...")
    full_model = build_full_model(base_model)
    
    # Разделение на train/val для дообучения
    X_short_train, X_short_val, Xm_short_train, Xm_short_val, y_short_train, y_short_val = train_test_split(
        X_short, Xm_short, y_short, test_size=0.2, random_state=42
    )
    
    full_history = full_model.fit(
        [X_short_train, Xm_short_train], y_short_train,
        validation_data=([X_short_val, Xm_short_val], y_short_val),
        epochs=CONFIG['epochs_fine_tune'],
        batch_size=CONFIG['batch_size'],
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Оценка
    print("\nEvaluating models...")
    # На полных данных (только базовая модель)
    base_mae_full = base_model.evaluate(X_test_full, y_test_full, verbose=0)
    print(f"Base model MAE (full data): {y_scaler.inverse_transform([[base_mae_full]])[0][0]:.2f}°C")
    
    # На коротких данных (сравнение базовой и полной)
    base_mae_short = base_model.evaluate(X_short_val, y_short_val, verbose=0)
    full_mae_short = full_model.evaluate([X_short_val, Xm_short_val], y_short_val, verbose=0)
    
    print(f"Base model MAE (short data): {y_scaler.inverse_transform([[base_mae_short]])[0][0]:.2f}°C")
    print(f"Full model MAE (short data): {y_scaler.inverse_transform([[full_mae_short]])[0][0]:.2f}°C")
    print(f"Improvement: {(base_mae_short - full_mae_short)/base_mae_short:.1%}")

if __name__ == "__main__":
    main()
