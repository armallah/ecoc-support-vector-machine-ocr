# src/data/make_dataset.py
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def fetch_emnist_data(input_dir='input', train_file='emnist-letters-train.csv', test_file='emnist-letters-test.csv'):
    """
    Load EMNIST data from input/*.csv
    """
    # Load training data
    train_df = pd.read_csv(os.path.join(input_dir, train_file), delimiter=",")
    train_df.dataframeName = train_file
    print(f"Training data shape: {train_df.shape}")
    
    # Load test data
    test_df = pd.read_csv(os.path.join(input_dir, test_file), delimiter=",")
    test_df.dataframeName = test_file
    print(f"Test data shape: {test_df.shape}")
    
    print("\nTraining data:")
    print(train_df.head())
    print("\nTest data:")
    print(test_df.head())
    
    return train_df, test_df

def visualize_sample(df, index=0):
    """
    Visualize a sample image from the dataset
    """
    img = df.iloc[index, 1:].values.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {df.iloc[index, 0]}")
    plt.axis('off')
    plt.show()

def save_data(df, filename, output_dir='../../data/raw'):
    """
    Save the dataframe as a numpy array
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, filename), df.values)
    print(f"Saved {filename} to {output_dir}")

if __name__ == "__main__":
    print("Files in input directory:")
    print(os.listdir("input"))
    
    train_df, test_df = fetch_emnist_data()
    
    # Visualize a sample from training data
    visualize_sample(train_df)