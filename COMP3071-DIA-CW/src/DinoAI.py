from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException


# Set options for the WebDriver.
options = Options()
# Turn off logging to keep terminal clean.
options.add_experimental_option('excludeSwitches', ['enable-logging'])
# Keep the browser running after the code finishes executing.
options.add_experimental_option("detach", True)

# Create a Service instance for running the ChromeDriver executable.
service = Service(executable_path=ChromeDriverManager().install())

# Create an instance of the Chrome WebDriver with the specified service and options. The driver object can be used to automate interactions with the Chrome browser.
driver = webdriver.Chrome(service=service, options=options)
driver.maximize_window()

try:
    # Navigate to the Chrome Dino website.
    driver.get("chrome://dino/")
except WebDriverException as e:
    # Ignore "ERR_INTERNET_DISCONNECTED" error thrown because this game is available offline.
    if "ERR_INTERNET_DISCONNECTED" in str(e):
        pass  # Ignore the exception.
    else:
        raise e  # Handle other WebDriverExceptions.
