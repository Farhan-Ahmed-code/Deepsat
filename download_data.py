import os
import zipfile

zip_path = "EuroSAT.zip"
extract_dir = "./EuroSAT_Dataset"
target_data_folder = os.path.join(extract_dir, "2750")

print("Checking for Kaggle EuroSAT.zip...")

if os.path.exists(zip_path):
    print("Zip file found! Extracting images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete!")
    
    # Kaggle sometimes nests the folders. Let's make sure they are in the '2750' directory
    # so our data loader code doesn't break.
    extracted_contents = os.listdir(extract_dir)
    
    if "EuroSAT" in extracted_contents and not os.path.exists(target_data_folder):
        print("Reorganizing Kaggle folder structure...")
        os.rename(os.path.join(extract_dir, "EuroSAT"), target_data_folder)
        
    if os.path.exists(target_data_folder):
        classes = os.listdir(target_data_folder)
        print(f"\nSuccess! Found {len(classes)} classes ready for ResNet-50:")
        print(classes)
    else:
        print("\nFolders extracted, but please check the EuroSAT_Dataset directory to verify the structure.")
else:
    print("Error: Could not find EuroSAT.zip.")
    print("Please make sure you moved the downloaded zip from Kaggle into the DeepSatVision folder!")