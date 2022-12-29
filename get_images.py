import os
import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve

# Set the search term and number of images to download
search_term = "rihanna"
num_images = 50

# Set the base URL for the Google Image search
base_url = "https://www.google.com/search?q=" + search_term + "&source=lnms&tbm=isch"

# Send the search request and get the response
response = requests.get(base_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all image tags
    image_tags = soup.find_all("img")

    # Set the counter for the image names
    counter = 0

    # Create a folder for the images
    if not os.path.exists(search_term):
        os.makedirs(search_term)

    # Iterate through the image tags and download the images
    for image_tag in image_tags:
        # Get the image URL
        image_url = image_tag.get("src")

        # Check if the URL is valid
        if image_url.startswith("http"):
            # Download the image and save it to the folder
            urlretrieve(image_url, str(search_term + f"/{search_term}" + str(counter) + ".png"))
            counter += 1

            # Check if we have reached the required number of images
            if counter > num_images:
                break

else:
    # Request was not successful
    print("Error fetching search results")
