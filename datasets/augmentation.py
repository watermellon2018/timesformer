import albumentations as A

def get_train_augmentation(config):
    transform = A.Compose([
        A.Resize(config['img_size'], config['img_size'])
    ])

    return transform