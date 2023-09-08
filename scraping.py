import os
import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image


num_images_to_scrape_train = 100
num_images_to_scrape_test = 50
images_path = 'images/'
test_images_path = 'test/'


# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_directory(images_path)


# Function to filter images (e.g., check if they contain a vehicle)
# def filter_images(image_directory, output_directory):
#     create_directory(output_directory)
#
#     for filename in os.listdir(image_directory):
#         image_path = os.path.join(image_directory, filename)
#         try:
#             # Open the image using Pillow (PIL)
#             img = Image.open(image_path)
#
#             # Implement your filtering logic here (e.g., using machine learning models)
#             # For simplicity, let's assume all downloaded images are relevant for now
#             img.save(os.path.join(output_directory, filename))
#             print()
#         except Exception as e:
#             print(f"Error processing {filename}: {str(e)}")

def load_train_images(image_directory):
    train_images = []
    for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)
        # Convert the image to a numpy array and add it to the list
        image_np = np.array(Image.open(image_path))
        train_images.append(image_np)
    return train_images

def scrape_images_for_class(class_string, num_images_to_scrape, image_directory):
    # Create a directory for the images
    # image_directory = os.path.join(images_path, class_string)
    create_directory(image_directory)

    webdriver_path = 'chromedriver'  # Replace with the actual path
    service = ChromeService(executable_path=webdriver_path)
    options = webdriver.ChromeOptions()
    options.binary_location = 'Chromium.app/Contents/MacOS/Chromium'
    options.add_argument("--start-maximized")  # Optional: Maximize the browser window
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get("https://www.google.com/imghp")

        # Find the search input element and enter the class string
        search_input = driver.find_element(By.NAME, "q")
        search_input.send_keys(f'{class_string} military')
        search_input.send_keys(Keys.RETURN)  # Simulate pressing Enter

        image_urls = []
        num_downloaded = 0

        scroll_pause_time = 2  # Adjust as needed

        while num_downloaded < num_images_to_scrape:
            # Scroll down to load more images
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            image_tags = soup.find_all('img')
            for img_tag in image_tags:
                if 'data-src' in img_tag.attrs:
                    image_url = img_tag['data-src']
                    if image_url not in image_urls:
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            content_length = int(response.headers.get("content-length", 0))
                            if content_length > 2000:
                                with open(os.path.join(image_directory, f"{num_downloaded}.jpg"), 'wb') as file:
                                    file.write(response.content)
                                num_downloaded += 1
                                if num_downloaded == num_images_to_scrape:
                                    break

        print(f'Downloaded {num_downloaded} images for {class_string}')

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        driver.quit()


if __name__ == "__main__":
    data_file_path = rf'data/WEO_Data_Sheet.xlsx'
    dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)
    fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']

    coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
    coarse_grain_classes = coarse_grain_results_df['Class Name'].values
    fine_grain_classes = {k: v for k, v in enumerate(fine_grain_results_df['Class Name'].values)}

    train_directories = {item for item in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, item))}
    classes_without_train_folder = sorted(list(set(fine_grain_classes.values()).difference(train_directories)))

    print(f'classes_without_folder: {len(classes_without_train_folder)}\n{classes_without_train_folder}')

    for c in classes_without_train_folder:
        scrape_images_for_class(c, num_images_to_scrape_train, os.path.join(images_path, c))

    test_directories = {item for item in os.listdir(images_path) if os.path.isdir(os.path.join(test_images_path, item))}
    classes_without_test_folder = sorted(list(set(fine_grain_classes.values()).difference(test_directories)))

    # Scrape additional 50 images for the test set
    for c in classes_without_test_folder:
        scrape_images_for_class(c, num_images_to_scrape_test, os.path.join(test_images_path, c))
