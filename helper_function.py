import os
import cv2
import pytesseract
import csv
import argparse
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_parts(image_path):
    """
    Extract digital text which is at top of image and handwritten text which is at bottom of image from original IAM dataset images.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width, _ = image.shape

    # Split image: top part for digital text, bottom for handwriting
    digital_crop = image[0:int(height * 0.18), :]     # top 18%
    handwritten_crop = image[int(height * 0.22):, :]  # from ~22% down

    # OCR digital text using Tesseract
    digital_pil = Image.fromarray(cv2.cvtColor(digital_crop, cv2.COLOR_BGR2RGB))
    digital_text = pytesseract.image_to_string(digital_pil).strip()

    return handwritten_crop, digital_text


def process_directory(input_dir, output_dir="handwriting_dataset"):
    """
    Processes all images in the input directory and saves handwritten images and label as text files as well as metadata as csv file.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, mode="w", newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["id", "image_path", "label_path"])

        for root, _, files in os.walk(input_dir):
            for filename in sorted(files):
                if not filename.lower().endswith(".png"):
                    continue

                file_id = os.path.splitext(filename)[0]
                image_path = os.path.join(root, filename)

                try:
                    handwritten_img, typed_text = extract_parts(image_path)

                    out_img_path = os.path.join(images_dir, f"{file_id}_handwritten.png")
                    out_txt_path = os.path.join(labels_dir, f"{file_id}.txt")

                    cv2.imwrite(out_img_path, handwritten_img)

                    with open(out_txt_path, "w", encoding="utf-8") as f:
                        f.write(typed_text)

                    # Metadata
                    csv_writer.writerow([file_id, out_img_path, out_txt_path])

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

    print(f"Processed data saved to: {output_dir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process IAM dataset directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing original IAM files.")
    parser.add_argument("--output_dir", type=str, default="handwriting_dataset", help="Directory to save handwritten and label data.")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)