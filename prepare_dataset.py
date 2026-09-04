import os
import shutil
import random

# Zip Extract செய்யப்பட்ட மூல ஃபோல்டர் பாதை
SOURCE_DIR = r"D:\deepfake_detection_system\Celeb-DF"  # உங்கள் Celeb-DF extraction folder பெயருக்கு ஏற்ப மாற்றவும்

# இலக்கு dataset பாதை
TARGET_DIR = r"D:\deepfake_detection_system\dataset"

# தேவைப்படும் கோப்பகங்களை உருவாக்குதல்
for split in ['train', 'val']:
    for category in ['real', 'fake']:
        os.makedirs(os.path.join(TARGET_DIR, split, category), exist_ok=True)

def process_and_copy(source_folder, category_name, train_ratio=0.8):
    if not os.path.exists(source_folder):
        print(f"Warning: Folder not found -> {source_folder}")
        return

    # படங்களின் பட்டியலைப் பெறுதல்
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy Train Images
    for img in train_images:
        src = os.path.join(source_folder, img)
        dst = os.path.join(TARGET_DIR, 'train', category_name, img)
        shutil.copy(src, dst)

    # Copy Validation Images
    for img in val_images:
        src = os.path.join(source_folder, img)
        dst = os.path.join(TARGET_DIR, 'val', category_name, img)
        shutil.copy(src, dst)

    print(f"Successfully processed {category_name}: {len(train_images)} Train, {len(val_images)} Validation images.")

# Celeb-DF ஃபோல்டர்களிலிருந்து படங்களை பிரித்து நகர்த்துதல்
# Celeb-real / YouTube-real ஆகியவை REAL படங்கள்; Celeb-synthesis என்பது FAKE படங்கள்
process_and_copy(os.path.join(SOURCE_DIR, "Celeb-real"), "real")
process_and_copy(os.path.join(SOURCE_DIR, "Celeb-synthesis"), "fake")

print("\nDataset preparation completed successfully!")