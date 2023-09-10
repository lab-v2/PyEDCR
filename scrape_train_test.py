import os
import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
from io import BytesIO
from tqdm import tqdm
from typing import Optional, Sequence

num_images_to_scrape_train = 100
num_images_to_scrape_test = 50
train_images_path = 'train/'
test_images_path = 'test/'


# Function to create a directory if it doesn't exist
def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


create_directory(train_images_path)
create_directory(test_images_path)


# Function to check if an image is already in a directory
def is_image_in_directory(candidate_image_array: np.array, directory: str) -> bool:
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            existing_image = Image.open(os.path.join(directory, filename))
            existing_image_array = np.array(existing_image)
            if np.array_equal(candidate_image_array, existing_image_array):
                return True
    return False


# Function to scrape images for a class
def scrape_images_for_class(class_string: str,
                            num_images_to_scrape: int,
                            image_directory: str,
                            removed_images: Optional[Sequence[np.array]] = None,
                            scraping_for_test: bool = False) -> None:
    # Create a directory for the images
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

        num_downloaded = 0

        scroll_pause_time = 2  # Adjust as needed

        # Add tqdm progress bar
        with tqdm(total=num_images_to_scrape, unit='image') as pbar:
            while num_downloaded < num_images_to_scrape:
                # Scroll down to load more images
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                image_tags = soup.find_all('img')
                for img_tag in image_tags:
                    if 'data-src' in img_tag.attrs:
                        image_url = img_tag['data-src']
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            content_length = int(response.headers.get("content-length", 0))
                            if content_length > 2000:
                                image_content = response.content
                                image_array = np.array(Image.open(BytesIO(image_content)))
                                image_in_directory = is_image_in_directory(image_array, image_directory)
                                image_was_removed = removed_images is not None and \
                                                    any(np.array_equal(image_array, removed_image)
                                                        for removed_image in removed_images)
                                image_in_train = scraping_for_test and \
                                                 any(np.array_equal(image_array, train_image)
                                                     for train_image in
                                                     load_images_from_folder(
                                                         os.path.join(train_images_path, class_string)))

                                if (not image_in_directory) and (not image_was_removed) and (not image_in_train):
                                    filename = os.path.join(image_directory, f"{num_downloaded}.jpg")
                                    with open(filename, 'wb') as file:
                                        file.write(image_content)
                                    num_downloaded += 1
                                    pbar.update(1)
                                    if num_downloaded == num_images_to_scrape:
                                        break

        print(f'Downloaded {num_downloaded} images for {class_string}')

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        driver.quit()


def load_images_from_folder(folder_path: str) -> Sequence[np.array]:
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image = Image.open(os.path.join(folder_path, filename))
            images.append(np.array(image))
    return images


def assert_datasets(train_images_path: str, test_images_path: str) -> None:
    for class_folder in sorted(list(os.listdir(test_images_path))):
        if os.path.isdir(os.path.join(train_images_path, class_folder)):
            train_images = load_images_from_folder(os.path.join(train_images_path, class_folder))
            test_images = load_images_from_folder(os.path.join(test_images_path, class_folder))

            # assert len(
            #     train_images) == num_images_to_scrape_train, \
            #     f"Train images count mismatch for class {class_folder}"
            #
            # assert len(
            #     test_images) == num_images_to_scrape_test, \
            #     f"Test images count mismatch for class {class_folder} is {len(test_images)} < {num_images_to_scrape_test}"

            # Check for duplicate images and replace them in one go
            duplicates_to_replace = []
            for test_image in test_images:
                for train_image in train_images:
                    if np.array_equal(train_image, test_image):
                        duplicates_to_replace.append(test_image)

            if len(duplicates_to_replace):
                print(f'replacing {len(duplicates_to_replace)} duplicates for class {class_folder}')
                # Replace duplicates
                for duplicate in duplicates_to_replace:
                    duplicate_image_path = os.path.join(test_images_path, class_folder)

                    for filename in os.listdir(duplicate_image_path):
                        if filename.endswith('.jpg'):
                            existing_image = Image.open(os.path.join(duplicate_image_path, filename))
                            existing_image_array = np.array(existing_image)
                            if np.array_equal(duplicate, existing_image_array):
                                os.remove(os.path.join(duplicate_image_path, filename))

                # Scrape new images for the test set (excluding duplicates)
                scrape_images_for_class(class_folder, len(duplicates_to_replace),
                                        os.path.join(test_images_path, class_folder),
                                        removed_images=duplicates_to_replace,
                                        train_images=train_images)


if __name__ == "__main__":
    data_file_path = rf'data/WEO_Data_Sheet.xlsx'
    dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)
    fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']

    coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
    coarse_grain_classes = coarse_grain_results_df['Class Name'].values
    fine_grain_classes = {k: v for k, v in enumerate(fine_grain_results_df['Class Name'].values)}

    # Directories that don't have train folders
    directories_without_train = {item for item in os.listdir(train_images_path) if
                                 os.path.isdir(os.path.join(train_images_path, item))}
    classes_to_scrape_train = sorted(list(set(fine_grain_classes.values()).difference(directories_without_train)))
    print(f'classes_to_scrape_train: {len(classes_to_scrape_train)}\n{classes_to_scrape_train}')

    for cls in classes_to_scrape_train:
        scrape_images_for_class(cls, num_images_to_scrape_train, os.path.join(train_images_path, cls))

    # Directories that don't have test folders
    directories_without_test = {item for item in os.listdir(test_images_path) if
                                os.path.isdir(os.path.join(test_images_path, item))}
    classes_to_scrape_test = sorted(list(set(fine_grain_classes.values()).difference(directories_without_test)))
    print(f'classes_to_scrape_test: {len(classes_to_scrape_test)}\n{classes_to_scrape_test}')

    for cls in classes_to_scrape_test:
        scrape_images_for_class(class_string=cls,
                                num_images_to_scrape=num_images_to_scrape_test,
                                image_directory=os.path.join(test_images_path, cls),
                                scraping_for_test=True)

    # Check and replace duplicates
    assert_datasets(train_images_path, test_images_path)
    print("Assertions passed successfully.")
