# main.py

from make_dataset import fetch_emnist_data, save_data
import preprocess

if __name__ == '__main__':
    # Fetch and save raw data
    train_df, test_df = fetch_emnist_data()
    save_data(train_df, 'emnist_train.npy')
    save_data(test_df, 'emnist_test.npy')
    
    # Preprocess data
    preprocess.main()
