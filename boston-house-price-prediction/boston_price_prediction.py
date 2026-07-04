# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
def load_dataset():
    """
    Load Boston Housing dataset
    """
    try:
        # URL for Boston Housing dataset (alternative source)
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)
        print("Dataset loaded successfully!")
        return df
    except:
        # If you have local file
        df = pd.read_csv('BostonHousing.csv')
        return df

# Data Exploration and Visualization
def explore_data(df):
    """
    Perform initial data exploration
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Distribution of target variable (MEDV - Median house price)
    plt.subplot(2, 2, 1)
    plt.hist(df['medv'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Median House Price ($1000s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of House Prices')
    
    # Correlation heatmap
    plt.subplot(2, 2, 2)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    
    # Important features vs target
    plt.subplot(2, 2, 3)
    plt.scatter(df['rm'], df['medv'], alpha=0.5)
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Median House Price ($1000s)')
    plt.title('Rooms vs House Price')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['lstat'], df['medv'], alpha=0.5)
    plt.xlabel('Lower Status Population (%)')
    plt.ylabel('Median House Price ($1000s)')
    plt.title('Lower Status Population vs House Price')
    
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

# Data Preprocessing
def preprocess_data(df):
    """
    Preprocess the data for modeling
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Separate features and target
    X = df.drop('medv', axis=1)
    y = df['medv']
    
    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nFeatures have been standardized")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train and Evaluate Models
def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models and evaluate them
    """
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Print results
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance for Random Forest
        if name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 5 Important Features:")
            print(feature_importance.head())
    
    return results

# Visualize Results
def visualize_results(results, y_test):
    """
    Visualize model predictions and performance
    """
    print("\n" + "="*50)
    print("VISUALIZATION OF RESULTS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (name, result) in enumerate(results.items()):
        if idx < 4:  # Only plot first 4 models
            row = idx // 2
            col = idx % 2
            
            ax = axes[row, col]
            
            # Actual vs Predicted scatter plot
            ax.scatter(y_test, result['predictions'], alpha=0.5, label='Predictions')
            
            # Perfect prediction line
            min_val = min(y_test.min(), result['predictions'].min())
            max_val = max(y_test.max(), result['predictions'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax.set_xlabel('Actual Prices ($1000s)')
            ax.set_ylabel('Predicted Prices ($1000s)')
            ax.set_title(f'{name}\nR² = {result["r2"]:.4f}, RMSE = {result["rmse"]:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Residual plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, (name, result) in enumerate(results.items()):
        if idx < 4:
            row = idx // 2
            col = idx % 2
            
            ax = axes[row, col]
            
            residuals = y_test - result['predictions']
            ax.scatter(result['predictions'], residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Prices ($1000s)')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{name} - Residual Plot')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Predict New Data
def predict_new_data(model, scaler, new_data):
    """
    Make predictions on new data
    """
    print("\n" + "="*50)
    print("PREDICTING NEW DATA")
    print("="*50)
    
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    
    print("\nPredictions (in $1000s):")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: ${pred:.2f}K")
    
    return predictions

# Main execution
def main():
    """
    Main function to run the complete pipeline
    """
    print("="*60)
    print("BOSTON HOUSE PRICE PREDICTION PROJECT")
    print("="*60)
    
    # Load dataset
    df = load_dataset()
    
    # Explore data
    correlation_matrix = explore_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train and evaluate models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Visualize results
    visualize_results(results, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best R² Score: {results[best_model_name]['r2']:.4f}")
    print(f"Best RMSE: {results[best_model_name]['rmse']:.2f}")
    
    # Save the model (optional)
    import joblib
    joblib.dump(best_model, 'best_boston_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nBest model saved as 'best_boston_model.pkl'")
    
    # Example: Predict on a new sample
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    # Create a sample (using mean values from dataset)
    sample_data = np.array([[
        df['crim'].mean(), df['zn'].mean(), df['indus'].mean(), 
        df['chas'].mean(), df['nox'].mean(), df['rm'].mean(),
        df['age'].mean(), df['dis'].mean(), df['rad'].mean(),
        df['tax'].mean(), df['ptratio'].mean(), df['black'].mean(),
        df['lstat'].mean()
    ]])
    
    sample_prediction = predict_new_data(best_model, scaler, sample_data)
    
    print("\n" + "="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()