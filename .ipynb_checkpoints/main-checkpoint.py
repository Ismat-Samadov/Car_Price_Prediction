import time
from bs4 import BeautifulSoup
from selenium import webdriver

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run Chrome in headless mode (without a visible browser window)
driver = webdriver.Chrome(options=options)

# Define the Lua script
lua_script = '''
    function main(splash, args)
      splash.private_mode_enabled = false
      url = args.url
      assert(splash:go(url))
      assert(splash:wait(2))  -- Wait for 2 seconds (adjust as needed)
      return splash:html()
    end
'''

# Function to scrape car details
def scrape_car_details(driver, url):
    driver.get(url)
    time.sleep(2)  # Add a sleep time for page to load (adjust as needed)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Extract car details
    phone = soup.find('a', class_='product-phones__list-i').get_text()
    product_title = soup.find('h1', class_='product-title').get_text()
    short_description = [item.get_text() for item in soup.select('h1.product-title span.nobr')]
    update_date = soup.find_all('li', class_='product-statistics__i-text')[0].get_text()
    views_count = soup.find_all('li', class_='product-statistics__i-text')[1].get_text()
    owner_name = soup.find('div', class_='product-owner__info-name').get_text()
    owner_region = soup.find('div', class_='product-owner__info-region').get_text()
    city = soup.select('span[itemprop="addressLocality"]')[0].get_text()
    brand = soup.select('span[itemprop="brand"] a')[0].get_text()
    model = soup.select('span[itemprop="model"] a')[0].get_text()
    year = soup.select('span[itemprop="vehicleModelDate"] a')[0].get_text()
    body_type = soup.select('span[itemprop="bodyType"]')[0].get_text()
    color = soup.select('span[itemprop="color"]')[0].get_text()
    engine = soup.select('span[itemprop="vehicleEngine"]')[0].get_text()
    mileage = soup.select('span[itemprop="mileageFromOdometer"]')[0].get_text()
    transmission = soup.select('span[itemprop="vehicleTransmission"]')[0].get_text()
    drive_train = soup.select('span[itemprop="driveWheelConfiguration"]')[0].get_text()
    is_new = soup.select('span[itemprop="isNew"]')[0].get_text()
    number_of_seats = soup.select('span[itemprop="seatingCapacity"]')[0].get_text()
    number_of_prior_owners = soup.select('span[itemprop="numberOfPreviousOwners"]')[0].get_text()
    condition = soup.select('span[itemprop="itemCondition"]')[0].get_text()
    market = soup.select('span[itemprop="market"]')[0].get_text()

    car_details = {
        'short_description': short_description,
        'product_title': product_title,
        'update_date': update_date,
        'views_count': views_count,
        'owner_phone': phone,
        'owner_name': owner_name,
        'owner_region': owner_region,
        'city': city,
        'brand': brand,
        'model': model,
        'year': year,
        'body_type': body_type,
        'color': color,
        'engine': engine,
        'mileage': mileage,
        'transmission': transmission,
        'drive_train': drive_train,
        'is_new': is_new,
        'number_of_seats': number_of_seats,
        'number_of_prior_owners': number_of_prior_owners,
        'condition': condition,
        'market': market
    }

    return car_details

# Start scraping
base_url = "https://turbo.az/autos?page="
page_number = 1

while True:
    url = base_url + str(page_number)
    driver.get(url)

    # Render the page using Lua script
    driver.execute_script(lua_script)

    # Extract car links
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    car_links = [a['href'] for a in soup.select('div.product-properties__i a[href]')]

    if not car_links:
        break  # No more car links found, exit the loop

    for car_link in car_links:
        car_details = scrape_car_details(driver, car_link)
        print(car_details)

    page_number += 1

# Close the WebDriver
driver.quit()
