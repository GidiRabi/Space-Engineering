import cv2
import os

def compress_image(input_path, output_path=None, quality=80):
    """
    Compress an image using JPEG compression.
    Args:
        input_path (str): Path to the input image file.
        output_path (str, optional): Path to save the compressed image. 
                                     If None, saves as input_path + '_compressed.jpg'.
        quality (int): JPEG quality (0-100, higher means better quality and less compression).
    Returns:
        str: Path to the compressed image.
    """
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read input image: {input_path}")

    # Set output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_compressed.jpg"

    # Compress and save the image
    success = cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise IOError(f"Could not write compressed image to: {output_path}")

    return output_path

if __name__ == "__main__":
    # Example usage: compress an image provided by the user
    input_path = input("Enter path to image to compress: ").strip()
    quality = input("Enter JPEG quality (0-100, default 80): ").strip()
    quality = int(quality) if quality.isdigit() else 80

    try:
        output_path = compress_image(input_path, quality=quality)
        print(f"Compressed image saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")