"""Linear Regression Model Implementation."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class LinearRegressionModel:
    """
    A class to implement Linear Regression using scikit-learn.
    
    Attributes:
        model: LinearRegression model from scikit-learn
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
    """
    
    def __init__(self):
        """Initialize the Linear Regression model."""
        self.model = LinearRegression()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.is_trained = False
    
    def load_data(self, filepath, target_column='y', test_size=0.2, random_state=42):
        """
        Load data from a CSV file and split into train/test sets.
        
        Args:
            filepath (str): Path to the CSV file
            target_column (str): Name of the target variable column
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self, X_train=None, y_train=None):
        """
        Train the linear regression model.
        
        Args:
            X_train: Training features (optional, uses loaded data if None)
            y_train: Training target (optional, uses loaded data if None)
        
        Returns:
            self: The trained model
        """
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.y_train = y_train
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Load data first or provide X_train and y_train.")
        
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
        
        Returns:
            array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X=None, y=None):
        """
        Evaluate the model performance.
        
        Args:
            X: Features to evaluate on (optional, uses test data if None)
            y: True values (optional, uses test data if None)
        
        Returns:
            dict: Dictionary containing MSE, RMSE, and R2 score
        """
        if X is None or y is None:
            if self.X_test is None or self.y_test is None:
                raise ValueError("No test data available. Provide X and y or load data first.")
            X = self.X_test
            y = self.y_test
        
        predictions = self.predict(X)
        
        mse = mean_squared_error(y, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    def get_coefficients(self):
        """
        Get the model coefficients and intercept.
        
        Returns:
            dict: Dictionary with coefficients and intercept
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }
    
    def plot_predictions(self, X=None, y=None, save_path=None):
        """
        Plot actual vs predicted values.
        
        Args:
            X: Features to predict on (optional, uses test data if None)
            y: True values (optional, uses test data if None)
            save_path (str): Path to save the plot (optional)
        """
        if X is None or y is None:
            if self.X_test is None or self.y_test is None:
                raise ValueError("No test data available. Provide X and y or load data first.")
            X = self.X_test
            y = self.y_test
        
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.6, edgecolors='k')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals(self, X=None, y=None, save_path=None):
        """
        Plot residuals (errors) of predictions.
        
        Args:
            X: Features to predict on (optional, uses test data if None)
            y: True values (optional, uses test data if None)
            save_path (str): Path to save the plot (optional)
        """
        if X is None or y is None:
            if self.X_test is None or self.y_test is None:
                raise ValueError("No test data available. Provide X and y or load data first.")
            X = self.X_test
            y = self.y_test
        
        predictions = self.predict(X)
        residuals = y - predictions
        
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.6, edgecolors='k')
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def summary(self):
        """
        Print a summary of the model.
        """
        if not self.is_trained:
            print("Model is not trained yet.")
            return
        
        coeffs = self.get_coefficients()
        metrics = self.evaluate()
        
        print("=" * 50)
        print("LINEAR REGRESSION MODEL SUMMARY")
        print("=" * 50)
        print(f"\nIntercept: {coeffs['intercept']:.6f}")
        print("\nCoefficients:")
        for i, coef in enumerate(coeffs['coefficients']):
            print(f"  Feature {i+1}: {coef:.6f}")
        print("\nModel Performance:")
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R²:   {metrics['r2']:.6f}")
        print("=" * 50)

