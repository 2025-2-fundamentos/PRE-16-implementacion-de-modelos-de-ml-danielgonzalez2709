"""Example usage of the Linear Regression Model."""

from homework.linear_regression import LinearRegressionModel


def main():
    """
    Main function to demonstrate the Linear Regression model.
    """
    print("=" * 60)
    print("LINEAR REGRESSION MODEL - EXAMPLE")
    print("=" * 60)
    
    # Initialize the model
    print("\n1. Initializing Linear Regression Model...")
    model = LinearRegressionModel()
    
    # Load data
    print("\n2. Loading data from 'files/input/data.csv'...")
    X_train, X_test, y_train, y_test = model.load_data(
        filepath='files/input/data.csv',
        target_column='y',
        test_size=0.2,
        random_state=42
    )
    
    print(f"   Training set size: {len(X_train)} samples")
    print(f"   Test set size: {len(X_test)} samples")
    print(f"   Number of features: {X_train.shape[1]}")
    
    # Train the model
    print("\n3. Training the model...")
    model.train()
    print("   Model trained successfully!")
    
    # Get coefficients
    print("\n4. Model Coefficients:")
    coeffs = model.get_coefficients()
    print(f"   Intercept: {coeffs['intercept']:.6f}")
    for i, coef in enumerate(coeffs['coefficients']):
        feature_name = X_train.columns[i]
        print(f"   {feature_name}: {coef:.6f}")
    
    # Evaluate the model
    print("\n5. Evaluating the model on test data...")
    metrics = model.evaluate()
    print(f"   Mean Squared Error (MSE): {metrics['mse']:.6f}")
    print(f"   Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
    print(f"   R² Score: {metrics['r2']:.6f}")
    
    # Make predictions on test data
    print("\n6. Making predictions on test data...")
    predictions = model.predict(X_test)
    print(f"   First 5 predictions: {predictions[:5]}")
    print(f"   First 5 actual values: {y_test.values[:5]}")
    
    # Print full summary
    print("\n7. Full Model Summary:")
    model.summary()
    
    # Plot results
    print("\n8. Generating plots...")
    try:
        print("   - Plotting Actual vs Predicted values...")
        model.plot_predictions()
        
        print("   - Plotting Residuals...")
        model.plot_residuals()
    except Exception as e:
        print(f"   Warning: Could not generate plots. Error: {e}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

