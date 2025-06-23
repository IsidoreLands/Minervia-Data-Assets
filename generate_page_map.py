#!/usr/bin/env python3
import csv
import os

def generate_mapping(num_png_files, output_csv_filename="png_to_original_page_map.csv"):
    """
    Generates a CSV file mapping generated PNG filenames to their original
    page numbers in the 'commentariinotar00schm.pdf'.

    The rule is:
    - PNG page-001.png corresponds to original page 125.
    - PNG page-002.png corresponds to original page 127.
    - So, original_page = 123 + (2 * png_file_number)
    """
    mapping = []
    header = ["png_filename", "original_page_number"]
    mapping.append(header)

    for i in range(1, num_png_files + 1):
        png_filename = f"page-{i:03d}.png"
        original_page_number = 123 + (2 * i)
        mapping.append([png_filename, original_page_number])

    try:
        with open(output_csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(mapping)
        print(f"Successfully generated mapping to '{output_csv_filename}' with {len(mapping)-1} entries.")
    except IOError as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    NUMBER_OF_PNGS = 134 # This should match the number of PNGs generated
    # Ensure the script is run from within the minervia_project directory
    # or adjust path for output_csv_filename if needed.
    # Example: output_path = os.path.join("path", "to", "minervia_project", "png_to_original_page_map.csv")
    generate_mapping(NUMBER_OF_PNGS)
