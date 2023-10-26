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
from typing import Sequence
import multiprocessing as mp
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data

from utils import create_directory

train_images_path = 'train/'
test_images_path = 'test/'

create_directory(train_images_path)
create_directory(test_images_path)


# Function to check if an image is already in a directory
def is_image_in_directory(candidate_image_array: np.array,
                          directory: str) -> bool:
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            existing_image = Image.open(os.path.join(directory, filename))
            existing_image_array = np.array(existing_image)
            if np.array_equal(candidate_image_array, existing_image_array):
                return True
    return False


# Function to scrape images for a class
def scrape_images_for_class(class_string: str,
                            image_directory: str,
                            scraping_for_test: bool = False) -> None:
    # Create a directory for the images

    print(f"Searching images for class {class_string}...")
    create_directory(image_directory)

    webdriver_path = 'chromedriver'  # Replace with the actual path
    service = ChromeService(executable_path=webdriver_path)

    options = webdriver.ChromeOptions()
    options.binary_location = 'Chromium.app/Contents/MacOS/Chromium'
    options.add_argument("--start-maximized")  # Optional: Maximize the browser window
    options.add_argument("--headless")  # Run in headless mode without a visible browser window
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get("https://www.google.com/imghp")

        # Find the search input element and enter the class string
        search_input = driver.find_element(By.NAME, "q")
        search_input.send_keys(f"{class_string} military"
                               f"{'' if class_string != 'Tornado' else ' multiple rocket launchers'}")
        search_input.send_keys(Keys.RETURN)  # Simulate pressing Enter

        scroll_pause_time = 1  # Adjust as needed

        # print('Scrolling all the way down...')
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                show_more_button = driver.find_element(By.XPATH, "//input[@value='Show more results']")
                if show_more_button.is_displayed():
                    show_more_button.click()
                    time.sleep(scroll_pause_time)  # Wait for more images to load
                else:
                    break
            last_height = new_height

        num_downloaded = 0

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        image_tags = soup.find_all('img')

        print(f"\nStarted scraping {'test' if scraping_for_test else 'train'} "
              f"images for class {class_string}...\n")

        with tqdm(total=len(image_tags), unit='image') as pbar:

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
                            image_in_train = scraping_for_test and any(np.array_equal(image_array, train_image)
                                                                       for train_image in
                                                                       load_images_from_folder(
                                                                           os.path.join(train_images_path,
                                                                                        class_string)))

                            if (not image_in_directory) and (not image_in_train):
                                filename = os.path.join(image_directory, f"{num_downloaded}.jpg")
                                with open(filename, 'wb') as file:
                                    file.write(image_content)
                                num_downloaded += 1
                                pbar.update(1)

        print(f'\nDownloaded {num_downloaded} images for {class_string}\n')

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


def plot_dataset_class_frequencies():
    # Define data transformations and loaders for train and test datasets
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root='train/', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root='test/', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Calculate class frequencies
    train_class_freq = [0] * len(train_dataset.classes)
    test_class_freq = [0] * len(test_dataset.classes)

    for _, target in train_loader.dataset.samples:
        train_class_freq[target] += 1

    for _, target in test_loader.dataset.samples:
        test_class_freq[target] += 1

    # Get class names
    class_names = train_dataset.classes

    # Plot class distributions with rotated labels
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(class_names))

    bar1 = ax.bar(index, train_class_freq, bar_width, label='Train')
    bar2 = ax.bar(index + bar_width, test_class_freq, bar_width, label='Test')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequency')
    ax.set_title('Class Distribution Comparison')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(class_names, rotation=90)  # Rotate labels 90 degrees
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_file_path = rf'../data/WEO_Data_Sheet.xlsx'
    dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)
    fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
    fine_grain_classes = set(fine_grain_results_df['Class Name'].values)

    # Directories that don't have train folders
    directories_without_train = {item for item in os.listdir(train_images_path) if
                                 os.path.isdir(os.path.join(train_images_path, item))}
    classes_to_scrape_train = sorted(list(fine_grain_classes.difference(directories_without_train)))
    print(f'classes_to_scrape_train: {len(classes_to_scrape_train)}\n{classes_to_scrape_train}')

    # Scraping train images
    pool = mp.Pool(processes=mp.cpu_count())
    pool.starmap(scrape_images_for_class,
                 [(cls, os.path.join(train_images_path, cls), False)
                  for cls in classes_to_scrape_train])
    pool.close()
    pool.join()

    # Directories that don't have test folders
    directories_without_test = {item for item in os.listdir(test_images_path) if
                                os.path.isdir(os.path.join(test_images_path, item))}
    classes_to_scrape_test = sorted(list(fine_grain_classes.difference(directories_without_test)))
    print(f'classes_to_scrape_test: {len(classes_to_scrape_test)}\n{classes_to_scrape_test}')

    # Scraping test images
    pool = mp.Pool(processes=mp.cpu_count())
    pool.starmap(scrape_images_for_class,
                 [(cls, os.path.join(test_images_path, cls), True)
                  for cls in classes_to_scrape_test])
    pool.close()
    pool.join()

    plot_dataset_class_frequencies()
