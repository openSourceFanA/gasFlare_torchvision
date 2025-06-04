from custom_dataset import CustomCocoDataset

def get_dataset(img_dir, mask_dir, ann_file):
    """
    Load custom dataset with masks.
    """
    dataset = CustomCocoDataset(img_dir, mask_dir, ann_file)
    return dataset
