from selenium import webdriver


_options = webdriver.ChromeOptions()
_options.add_argument('--headless')
_options.add_argument('--no-sandbox')
_options.add_argument('--disable-gpu')

driver = webdriver.Chrome(options=_options)
