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
import os
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directory for plots if not exists
os.makedirs('test_plots', exist_ok=True)

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
        for col in ['TEMPERATURE']:  # Add relevant metrics
            for window in [3, 7]:
                group[f'{col}_rolling_mean_{window}'] = group[col].rolling(window).mean()
                group[f'{col}_rolling_std_{window}'] = group[col].rolling(window).std()
        return group.fillna(method='bfill')

    data = data.groupby('OBJT_ID').apply(add_rolling_features).reset_index(drop=True)
    return data

# --- Sequence Creation with Proper Splitting ---
def create_sequences(data, obj_ids, features, target_col, window_size, forecast_horizon):
    X, y, ids = [], [], []

    # Pre-scale all data first
    feature_scaler = StandardScaler().fit(data[data['OBJT_ID'].isin(obj_ids)][features])
    target_scaler = StandardScaler().fit(data[data['OBJT_ID'].isin(obj_ids)][[target_col]])

    for obj_id in obj_ids:
        obj_data = data[data['OBJT_ID'] == obj_id].copy()
        if len(obj_data) < window_size + forecast_horizon:
            continue

        # Scale features and target
        scaled_features = feature_scaler.transform(obj_data[features])
        scaled_target = target_scaler.transform(obj_data[[target_col]])

        # Create sequences ensuring no future leakage
        for i in range(len(obj_data) - window_size - forecast_horizon):
            X.append(scaled_features[i:i+window_size])
            y.append(scaled_target[i+window_size:i+window_size+forecast_horizon])
            ids.append(obj_id)

    return np.array(X), np.array(y), np.array(ids), {'feature_scaler': feature_scaler, 'target_scaler': target_scaler}

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

    model.compile(optimizer=AdamW(learning_rate=0.01),
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

# --- Extract last N samples from each group for testing ---
def extract_test_samples(data, n_samples=5, window_size=7):
    # For each object, we need at least window_size + n_samples points
    test_samples = []
    train_samples = []
    
    for obj_id, group in data.groupby('OBJT_ID'):
        if len(group) >= window_size + n_samples:
            # Take last n_samples + window_size points
            test_samples.append(group.tail(n_samples + window_size))
            train_samples.append(group.iloc[:-(n_samples + window_size)])
        else:
            # If not enough points, skip this object for testing
            train_samples.append(group)
    
    train_data = pd.concat(train_samples, axis=0)
    test_data = pd.concat(test_samples, axis=0) if test_samples else pd.DataFrame()
    
    return train_data, test_data

# --- Prepare test sequences for rolling predictions ---
def prepare_test_sequences(test_data, features, target_col, window_size, forecast_horizon, scalers, obj_to_idx):
    X_test, y_test, ids_test = [], [], []
    
    for obj_id in test_data['OBJT_ID'].unique():
        obj_data = test_data[test_data['OBJT_ID'] == obj_id].copy()
        if len(obj_data) < window_size + 1:
            continue
            
        # Scale features and target
        scaled_features = scalers['feature_scaler'].transform(obj_data[features])
        scaled_target = scalers['target_scaler'].transform(obj_data[[target_col]])
        
        # Create rolling window predictions
        for i in range(len(obj_data) - window_size - forecast_horizon + 1):
            X_test.append(scaled_features[i:i+window_size])
            y_test.append(scaled_target[i+window_size:i+window_size+forecast_horizon])
            ids_test.append(obj_id)
    
    return np.array(X_test), np.array(y_test), np.array(ids_test)

# --- Evaluate on test samples with rolling predictions ---
def evaluate_test_samples(model, test_data, features, target_col, window_size, forecast_horizon, 
                         scalers, obj_to_idx, n_predictions=5):
    # Prepare test data
    X_test, y_test, ids_test = prepare_test_sequences(test_data, features, target_col, 
                                                     window_size, forecast_horizon, scalers, obj_to_idx)
    
    if len(X_test) == 0:
        print("Warning: No test samples were prepared. Check your window_size and forecast_horizon.")
        return pd.DataFrame(columns=['MAE', 'Min_AE', 'Max_AE', 'R2']), pd.DataFrame()

    # Convert object IDs to indices
    X_id_test = np.array([obj_to_idx[i] for i in ids_test]).reshape(-1, 1)

    # Predict
    try:
        y_pred = model.predict([X_test, X_id_test], batch_size=min(32, len(X_test)), verbose=1)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return pd.DataFrame(columns=['MAE', 'Min_AE', 'Max_AE', 'R2']), pd.DataFrame()

    # Inverse scaling
    y_true_real = scalers['target_scaler'].inverse_transform(y_test.reshape(-1, forecast_horizon))
    y_pred_real = scalers['target_scaler'].inverse_transform(y_pred.reshape(-1, forecast_horizon))

    # Create results DataFrame
    results = []
    for i, obj_id in enumerate(ids_test):
        results.append({
            'OBJT_ID': obj_id,
            'True_Value': y_true_real[i][0],
            'Predicted_Value': y_pred_real[i][0],
            'Absolute_Error': abs(y_true_real[i][0] - y_pred_real[i][0])
        })

    results_df = pd.DataFrame(results)
    
    # For each object, keep only the last n_predictions
    results_df = results_df.groupby('OBJT_ID').tail(n_predictions)

    # Calculate metrics per object
    if len(results_df) > 0:
        metrics = results_df.groupby('OBJT_ID').agg({
            'Absolute_Error': ['mean', 'min', 'max'],
            'True_Value': lambda x: r2_score(x, results_df.loc[x.index, 'Predicted_Value'])
        })
        metrics.columns = ['MAE', 'Min_AE', 'Max_AE', 'R2']
    else:
        metrics = pd.DataFrame(columns=['MAE', 'Min_AE', 'Max_AE', 'R2'])

    # Plot results for each object
    if test_data is not None and len(test_data) > 0:
        for obj_id in test_data['OBJT_ID'].unique():
            obj_data = test_data[test_data['OBJT_ID'] == obj_id]
            obj_results = results_df[results_df['OBJT_ID'] == obj_id]
            
            if len(obj_data) == 0 or len(obj_results) == 0:
                continue

            plt.figure(figsize=(10, 6))
            plt.plot(obj_data['DATE_STAMP'], obj_data[target_col], 'b-', label='True Values')
            
            # Get the dates for predictions (last n_predictions points)
            pred_dates = obj_data['DATE_STAMP'].iloc[-len(obj_results):]
            plt.plot(pred_dates, obj_results['Predicted_Value'], 'ro-', label='Predictions')

            if obj_id in metrics.index:
                plt.title(f'Temperature Forecast for Object {obj_id}\nMAE: {metrics.loc[obj_id, "MAE"]:.2f}, R2: {metrics.loc[obj_id, "R2"]:.2f}')
            else:
                plt.title(f'Temperature Forecast for Object {obj_id}')

            plt.xlabel('Date')
            plt.ylabel('Temperature')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plt.savefig(f'test_plots/object_{obj_id}_forecast.png')
            plt.close()

    return metrics, results_df

# --- Main Execution ---
def main():
    # Load and preprocess data
    data = pd.read_csv('/home/golovanks/ts/uadd-forecasting/research/experiments/golovan_0_create_dataset/data/processed/result_dataset.csv')
    data['volume'] = (data['BATH_HEIGHT'] / 100) * 6.9 * 4.12
    data = preprocess_data(data)

    # Parameters
    window_size = 7
    forecast_horizon = 1
    n_test_predictions = 5  # Number of predictions to make for each object

    # Extract test samples (last n_test_predictions + window_size points from each object)
    train_data, test_data = extract_test_samples(data, n_samples=n_test_predictions, window_size=window_size)
    print(f"Original data shape: {data.shape}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of unique objects in test: {test_data['OBJT_ID'].nunique() if not test_data.empty else 0}")

    # Feature selection
    base_features = ['BATH_HEIGHT', 'volume']
    rolling_features = [col for col in data.columns if 'rolling_' in col]
    time_features = ['day_of_week', 'day_of_year']
    features = base_features + rolling_features + time_features
    target_col = 'TEMPERATURE'

    # Prepare object mappings
    object_ids = train_data['OBJT_ID'].unique()
    num_objects = len(object_ids)
    obj_to_idx = {obj: i for i, obj in enumerate(object_ids)}
    train_data['OBJT_ID_IDX'] = train_data['OBJT_ID'].map(obj_to_idx)

    # Split objects into train/val (75/25)
    train_obj_ids, val_obj_ids = train_test_split(object_ids, test_size=0.25, random_state=42)

    # Create sequences with proper splitting
    X_train, y_train, ids_train, scalers_train = create_sequences(train_data, train_obj_ids, features, target_col, window_size, forecast_horizon)
    X_val, y_val, ids_val, scalers_val = create_sequences(train_data, val_obj_ids, features, target_col, window_size, forecast_horizon)

    # Prepare ID matrices
    X_id_train = np.array([obj_to_idx[i] for i in ids_train]).reshape(-1, 1)
    X_id_val = np.array([obj_to_idx[i] for i in ids_val]).reshape(-1, 1)

    # Build and train model
    model = build_improved_model(window_size, len(features), forecast_horizon, num_objects)
    model.summary()

    train_data = (X_train, X_id_train, y_train)
    val_data = (X_val, X_id_val, y_val)
    history = train_model(model, train_data, val_data, epochs=150)

    # Evaluate on test samples with rolling predictions
    test_metrics, test_results = evaluate_test_samples(model, test_data, features, target_col, 
                                                     window_size, forecast_horizon, 
                                                     scalers_train, obj_to_idx, 
                                                     n_predictions=n_test_predictions)

    # Print and save metrics
    print("\nTest Metrics per Object:")
    print(test_metrics)

    if not test_metrics.empty:
        print("\nOverall Test Metrics:")
        print(f"Mean MAE: {test_metrics['MAE'].mean():.3f}")
        print(f"Mean R2: {test_metrics['R2'].mean():.3f}")
        print(f"Min MAE: {test_metrics['MAE'].min():.3f}")
        print(f"Max MAE: {test_metrics['MAE'].max():.3f}")

        # Save metrics to CSV
        test_metrics.to_csv('test_plots/test_metrics.csv')
        test_results.to_csv('test_plots/test_results.csv', index=False)
    else:
        print("\nNo test metrics to report - no valid test samples were processed.")

if __name__ == "__main__":
    main()
