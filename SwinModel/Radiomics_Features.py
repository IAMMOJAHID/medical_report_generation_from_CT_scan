import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import os

def extract_radiomics_features(image_path, mask_path, params_path="Params.yaml"):
    """
    Extracts radiomics features from a medical image and its corresponding mask.

    Args:
        image_path (str): Path to the medical image file (e.g., NIfTI, DICOM).
        mask_path (str): Path to the segmentation mask file.
        params_path (str, optional): Path to the parameter file. Defaults to "Params.yaml".

    Returns:
        dict: A dictionary containing the extracted radiomics features.
    """
    try:
        # Read the image and mask using SimpleITK
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        # Initialize the feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(params_path)

        # Enable all features (or specify specific feature classes in the params file)
        #extractor.enableAllFeatures() #This is enabled by default.

        # Extract features
        feature_vector = extractor.execute(image, mask)

        return feature_vector

    except Exception as e:
        print(f"Error extracting radiomics features: {e}")
        return None

#Example Usage. Create dummy files or replace with your actual data paths.
#Create dummy files if they don't exist, for testing purposes.
def create_dummy_files(image_path, mask_path):
    if not os.path.exists(image_path):
        dummy_image = sitk.Image([64, 64, 64], sitk.sitkInt16)
        sitk.WriteImage(dummy_image, image_path)
    if not os.path.exists(mask_path):
        dummy_mask = sitk.Image([64, 64, 64], sitk.sitkUInt8)
        sitk.WriteImage(dummy_mask, mask_path)

if __name__ == "__main__":
    image_file = "dummy_image.nii.gz"
    mask_file = "dummy_mask.nii.gz"

    create_dummy_files(image_file, mask_file) #creates dummy files for testing

    # Create a dummy parameter file (Params.yaml) if needed, or use a custom one.
    # A basic Params.yaml can be created with a text editor and saved.
    # Example Params.yaml:
    # imageType: Original
    # featureClass:
    #   shape: {}
    #   firstorder: {}
    #   glcm: {}
    #   glrlm: {}
    #   glszm: {}
    #   gldm: {}
    #   ngtdm: {}
    # voxelSetting:
    #   binWidth: 25

    if not os.path.exists("Params.yaml"):
      with open("Params.yaml", "w") as f:
        f.write("imageType: Original\n")
        f.write("featureClass:\n")
        f.write("  shape: {}\n")
        f.write("  firstorder: {}\n")
        f.write("  glcm: {}\n")
        f.write("  glrlm: {}\n")
        f.write("  glszm: {}\n")
        f.write("  gldm: {}\n")
        f.write("  ngtdm: {}\n")
        f.write("voxelSetting:\n")
        f.write("  binWidth: 25\n")


    features = extract_radiomics_features(image_file, mask_file)

    if features:
        for key, value in features.items():
            print(f"{key}: {value}")