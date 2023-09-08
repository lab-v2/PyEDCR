import os
import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

# from PIL import Image

num_images_to_scrape = 100
images_path = 'images/'


# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_directory(images_path)


# Function to download an image from a URL
def download_image(url, directory, filename):
    response = requests.get(url)
    if response.status_code == 200:
        content_length = int(response.headers.get("content-length", 0))
        # Check if the image size is larger than 2 KB (adjust as needed)
        if content_length > 2000:
            with open(os.path.join(directory, filename), 'wb') as file:
                file.write(response.content)


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


def scrape_images_for_class(class_string):
    # Create a directory for the images
    image_directory = os.path.join(images_path, class_string)
    create_directory(image_directory)

    # Create a directory for the filtered images
    # filtered_directory = os.path.join('filtered_images', class_string)
    # create_directory(filtered_directory)

    webdriver_path = 'chromedriver'  # Replace with the actual path
    service = ChromeService(executable_path=webdriver_path)
    options = webdriver.ChromeOptions()
    options.binary_location = 'Chromium.app/Contents/MacOS/Chromium'
    options.add_argument("--start-maximized")  # Optional: Maximize the browser window
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Specify the path to the ChromeDriver executable

        driver.get("https://www.google.com/imghp")

        # Find the search input element and enter the class string
        # Find the search input element by name and enter your query
        search_input = driver.find_element(By.NAME, "q")
        search_input.send_keys(f'{class_string} military')
        search_input.send_keys(Keys.RETURN)  # Simulate pressing Enter

        # Perform additional interactions to load more images (scroll, click "Load more," etc.)
        # You may need to adjust this part based on Google Images' current structure
        # This may also depend on the language and region settings of your browser

        # Extract image URLs from the search results
        # This part may involve parsing the HTML, which can be complex
        # You can use BeautifulSoup to help with HTML parsing
        image_urls = []

        # Keep scrolling to load more images until a certain number is reached (e.g., 100 images)

        scroll_pause_time = 2  # Adjust as needed

        while len(image_urls) < num_images_to_scrape:
            # Scroll down to load more images
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)

            # Extract image URLs from the updated page source
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            image_tags = soup.find_all('img')
            for img_tag in image_tags:
                if 'data-src' in img_tag.attrs:
                    image_url = img_tag['data-src']
                    if image_url not in image_urls:
                        image_urls.append(image_url)

        # Download the images using the extracted URLs
        for i, image_url in enumerate(image_urls):
            download_image(image_url, image_directory, f"{i}.jpg")
            time.sleep(0.1)  # Add a delay to avoid overloading the server

        # Filter the downloaded images (e.g., check if they contain a vehicle)
        # filter_images(image_directory, filtered_directory)

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

    directories = {item for item in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, item))}
    classes_without_folder = sorted(list(set(fine_grain_classes.values()).difference(directories)))

    print(f'classes_without_folder: {len(classes_without_folder)}\n{classes_without_folder}')

    for c in classes_without_folder:
        scrape_images_for_class(c)
