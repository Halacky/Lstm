import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding, Reshape, BatchNormalization, Dropout, Layer
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Enhanced Attention Layer with Global Context ---
class ContextAwareAttention(Layer):
    def __init__(self, units=64, **kwargs):
        super(ContextAwareAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Input shape: [(batch, timesteps, features), (batch, context_features)]
        self.W1 = self.add_weight(name='attention_weights1',
                                shape=(input_shape[0][-1] + input_shape[1][-1], self.units),
                                initializer='glorot_uniform',
                                trainable=True)
        self.W2 = self.add_weight(name='attention_weights2',
                                shape=(self.units, 1),
                                initializer='glorot_uniform',
                                trainable=True)
        super(ContextAwareAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs: [sequence_input, context_input]
        sequence, context = inputs
        
        # Expand context to match timesteps
        context = tf.expand_dims(context, axis=1)
        context = tf.tile(context, [1, tf.shape(sequence)[1], 1])
        
        # Combine sequence and context
        combined = tf.concat([sequence, context], axis=-1)
        
        # Compute attention scores
        e = tf.tanh(tf.matmul(combined, self.W1))
        e = tf.matmul(e, self.W2)
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention
        output = sequence * a
        return tf.reduce_sum(output, axis=1)

# --- Data Preprocessing with Feature Engineering ---
def preprocess_data(data):
    data['DATE_STAMP'] = pd.to_datetime(data['DATE_STAMP'])
    data.sort_values(by=['OBJT_ID', 'DATE_STAMP'], inplace=True)
    
    # Add time-based features
    data['day_of_week'] = data['DATE_STAMP'].dt.dayofweek
    data['day_of_year'] = data['DATE_STAMP'].dt.dayofyear
    
    # Group-wise feature engineering
    def add_rolling_features(group):
        group = group.sort_values('DATE_STAMP')
        for col in ['TEMPERATURE', 'VOLTAGE']:  # Add relevant metrics
            for window in [3, 7]:
                group[f'{col}_rolling_mean_{window}'] = group[col].rolling(window).mean()
                group[f'{col}_rolling_std_{window}'] = group[col].rolling(window).std()
        return group.fillna(method='bfill')
    
    data = data.groupby('OBJT_ID').apply(add_rolling_features).reset_index(drop=True)
    return data

# --- Sequence Creation with Proper Splitting ---
def create_sequences(data, obj_ids, features, target_col, window_size, forecast_horizon, shuffle_within_obj=False):
    X, y, ids = [], [], []
    scalers = {}
    
    for obj_id in obj_ids:
        obj_data = data[data['OBJT_ID'] == obj_id].copy()
        if len(obj_data) < window_size + forecast_horizon:
            continue
        
        # Use single scaler for all objects for consistent scaling
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        scaled_features = feature_scaler.fit_transform(obj_data[features])
        scaled_target = target_scaler.fit_transform(obj_data[[target_col]])
        
        scalers[obj_id] = (feature_scaler, target_scaler)
        
        indices = list(range(len(obj_data) - window_size - forecast_horizon))
        if shuffle_within_obj:
            np.random.shuffle(indices)

        for i in indices:
            X.append(scaled_features[i:i+window_size])
            y.append(scaled_target[i+window_size:i+window_size+forecast_horizon])
            ids.append(obj_id)
    
    return np.array(X), np.array(y), np.array(ids), scalers

# --- Improved Model Architecture ---
def build_improved_model(window_size, num_features, forecast_horizon, num_objects, embedding_dim=8):
    # Input layers
    input_seq = Input(shape=(window_size, num_features), name='input_seq')
    obj_id_input = Input(shape=(1,), name='id_input')
    
    # Object embedding with richer representation
    obj_embed = Embedding(input_dim=num_objects, 
                         output_dim=embedding_dim, 
                         name='object_embedding')(obj_id_input)
    obj_embed = Reshape((embedding_dim,))(obj_embed)
    
    # Context processing branch
    context_branch = Dense(16, activation='relu')(obj_embed)
    context_branch = BatchNormalization()(context_branch)
    
    # Temporal processing branch
    temporal_branch = LSTM(32, return_sequences=True)(input_seq)
    temporal_branch = BatchNormalization()(temporal_branch)
    temporal_branch = Dropout(0.3)(temporal_branch)
    
    # Context-aware attention
    attention_out = ContextAwareAttention(units=32)([temporal_branch, context_branch])
    
    # Combine features
    combined = Concatenate()([attention_out, context_branch])
    
    # Prediction head
    x = Dense(32, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    main_output = Dense(forecast_horizon, name='main_output')(x)
    
    # Object-specific deviation (bias terms)
    deviation_layer = Embedding(input_dim=num_objects, 
                              output_dim=forecast_horizon, 
                              name="deviation_lookup")(obj_id_input)
    deviation_layer = Reshape((forecast_horizon,))(deviation_layer)
    
    # Final output with object-specific adjustments
    final_output = tf.keras.layers.Add(name='final_output')([main_output, deviation_layer])
    
    return Model(inputs=[input_seq, obj_id_input], outputs=final_output)

# --- Two-phase Training Approach ---
def train_model(model, train_data, val_data, epochs=100, batch_size=64):
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]
    
    # Phase 1: Train with frozen temporal layers (focus on global patterns)
    for layer in model.layers:
        if 'lstm' in layer.name.lower() or 'attention' in layer.name.lower():
            layer.trainable = False
    
    model.compile(optimizer=AdamW(learning_rate=0.01),
                 loss='mae',
                 metrics=['mae', 'mse'])
    
    print("\nPhase 1: Training global patterns")
    model.fit(
        [train_data[0], train_data[1]], train_data[2],
        validation_data=([val_data[0], val_data[1]], val_data[2]),
        epochs=int(epochs*0.3),
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Full model training
    for layer in model.layers:
        layer.trainable = True
    
    model.compile(optimizer=AdamW(learning_rate=0.001),
                 loss='mae',
                 metrics=['mae', 'mse'])
    
    print("\nPhase 2: Full model training")
    history = model.fit(
        [train_data[0], train_data[1]], train_data[2],
        validation_data=([val_data[0], val_data[1]], val_data[2]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# --- Enhanced Evaluation ---
def evaluate_model(model, X_test, X_id_test, y_test, scalers, obj_to_idx):
    # Predict
    y_pred = model.predict([X_test, X_id_test])
    
    # Inverse scaling
    y_true_real, y_pred_real = [], []
    obj_ids = [list(obj_to_idx.keys())[list(obj_to_idx.values()).index(idx[0])] for idx in X_id_test]
    
    for i in range(len(y_test)):
        obj_id = obj_ids[i]
        _, target_scaler = scalers[obj_id]
        y_true_real.append(target_scaler.inverse_transform(y_test[i].reshape(1, -1)))
        y_pred_real.append(target_scaler.inverse_transform(y_pred[i].reshape(1, -1)))
    
    y_true_real = np.array(y_true_real).flatten()
    y_pred_real = np.array(y_pred_real).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    r2 = r2_score(y_true_real, y_pred_real)
    
    print(f'Overall Metrics:')
    print(f'MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}')
    
    # Per-object metrics
    obj_metrics = []
    for obj_id in np.unique(obj_ids):
        mask = np.array(obj_ids) == obj_id
        if sum(mask) == 0:
            continue
        obj_mae = mean_absolute_error(y_true_real[mask], y_pred_real[mask])
        obj_metrics.append({'OBJT_ID': obj_id, 'MAE': obj_mae})
    
    metrics_df = pd.DataFrame(obj_metrics)
    print("\nPer-object MAE statistics:")
    print(metrics_df['MAE'].describe())
    
    return y_true_real, y_pred_real, metrics_df

# --- Visualization ---
def plot_results(y_true, y_pred, metrics_df, n_samples=200):
    plt.figure(figsize=(18, 12))
    
    # True vs Predicted
    plt.subplot(2, 2, 1)
    plt.plot(y_true[:n_samples], label='True')
    plt.plot(y_pred[:n_samples], label='Predicted')
    plt.title('True vs Predicted (First {} Samples)'.format(n_samples))
    plt.xlabel('Sample Index')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    
    # Error distribution
    plt.subplot(2, 2, 2)
    errors = np.abs(y_true - y_pred)
    sns.histplot(errors, bins=30, kde=True)
    plt.title('Absolute Error Distribution')
    plt.xlabel('Absolute Error')
    
    # Per-object performance
    plt.subplot(2, 2, 3)
    sns.boxplot(x='MAE', data=metrics_df)
    plt.title('MAE Distribution Across Objects')
    
    # Scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Scatter')
    
    plt.tight_layout()
    plt.savefig('temperature_forecast_results.png')
    plt.show()

# --- Main Execution ---
def main():
    # Load and preprocess data
    data = pd.read_csv('/home/golovanks/ts/uadd-forecasting/research/experiments/golovan_0_create_dataset/data/processed/result_dataset.csv')
    data['volume'] = (data['BATH_HEIGHT'] / 100) * 6.9 * 4.12
    data = preprocess_data(data)
    
    # Feature selection
    base_features = ['VOLTAGE', 'BATH_HEIGHT', 'volume']
    rolling_features = [col for col in data.columns if 'rolling_' in col]
    time_features = ['day_of_week', 'day_of_year']
    features = base_features + rolling_features + time_features
    target_col = 'TEMPERATURE'
    
    window_size, forecast_horizon = 7, 1
    
    # Prepare object mappings
    object_ids = data['OBJT_ID'].unique()
    num_objects = len(object_ids)
    obj_to_idx = {obj: i for i, obj in enumerate(object_ids)}
    data['OBJT_ID_IDX'] = data['OBJT_ID'].map(obj_to_idx)
    
    # Split objects into train/val/test (60/20/20)
    train_obj_ids, test_obj_ids = train_test_split(object_ids, test_size=0.2, random_state=42)
    train_obj_ids, val_obj_ids = train_test_split(train_obj_ids, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    # Create sequences with proper splitting
    X_train, y_train, ids_train, scalers_train = create_sequences(data, train_obj_ids, features, target_col, window_size, forecast_horizon)
    X_val, y_val, ids_val, scalers_val = create_sequences(data, val_obj_ids, features, target_col, window_size, forecast_horizon)
    X_test, y_test, ids_test, scalers_test = create_sequences(data, test_obj_ids, features, target_col, window_size, forecast_horizon)
    
    # Prepare ID matrices
    X_id_train = np.array([obj_to_idx[i] for i in ids_train]).reshape(-1, 1)
    X_id_val = np.array([obj_to_idx[i] for i in ids_val]).reshape(-1, 1)
    X_id_test = np.array([obj_to_idx[i] for i in ids_test]).reshape(-1, 1)
    
    # Build and train model
    model = build_improved_model(window_size, len(features), forecast_horizon, num_objects)
    model.summary()
    
    train_data = (X_train, X_id_train, y_train)
    val_data = (X_val, X_id_val, y_val)
    history = train_model(model, train_data, val_data, epochs=150)
    
    # Evaluate
    y_true, y_pred, metrics_df = evaluate_model(model, X_test, X_id_test, y_test, {**scalers_train, **scalers_val, **scalers_test}, obj_to_idx)
    plot_results(y_true, y_pred, metrics_df)

if __name__ == "__main__":
    main()
