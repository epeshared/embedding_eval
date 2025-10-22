# Download archives from the repository releases
print("Downloading Flickr8K dataset...")
!wget -q https://github.com/Avaneesh40585/Flickr8k-Dataset/releases/download/v1.0/Flickr8k_Dataset.zip  # Images
!wget -q https://github.com/Avaneesh40585/Flickr8k-Dataset/releases/download/v1.0/Flickr8k_text.zip     # Text annotations

# Extract to current directory
print("Extracting dataset...")
!unzip -qq Flickr8k_Dataset.zip
!unzip -qq Flickr8k_text.zip

# Clean up to save disk space
print("Cleaning up zip files...")
!rm Flickr8k_Dataset.zip Flickr8k_text.zip

print("Dataset setup complete!")
