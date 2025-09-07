import json
import os

from sklearn.model_selection import train_test_split

from burn_scar_detection import config


def create_and_save_splits(seed=config.RANDOM_SEED):
    """Generates train/validation splits from file IDs and saves them."""
    file_ids = sorted(
        [
            f.replace('.tif', '')
            for f in os.listdir(config.RAW_T1_DIR)
            if f.endswith('.tif')
        ]
    )

    train_ids, val_ids = train_test_split(file_ids, test_size=0.2, random_state=seed)

    split_data = {'train': train_ids, 'validation': val_ids}

    split_file_path = os.path.join(config.PROCESSED_DATA_DIR, 'splits.json')
    os.makedirs(os.path.dirname(split_file_path), exist_ok=True)

    with open(split_file_path, 'w') as f:
        json.dump(split_data, f, indent=4)

    print(
        f'Data split created: {len(train_ids)} train samples, {len(val_ids)} validation samples.'
    )
    print(f'Split definition saved to {split_file_path}')


if __name__ == '__main__':
    create_and_save_splits()
