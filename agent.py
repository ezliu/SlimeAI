#    def create_driver(self):
#        """Create a driver"""
#        assert not hasattr(self, 'driver'), \
#                'Instance {} already has a driver'.format(self.index)
#        options = webdriver.ChromeOptions()
#        if self.headless:
#            options.add_argument('headless')
#            options.add_argument('disable-gpu')
#            options.add_argument('no-sandbox')
#        else:
#            options.add_argument('app=' + self.url)
#            options.add_argument('window-size={},{}'
#                    .format(self.window_width, self.window_height))
#            options.add_argument('window-position={},{}'
#                    .format(9000, 30 + self.index * (self.window_height + 30)))
#        self.driver = webdriver.Chrome(chrome_options=options)
#        self.driver.implicitly_wait(5)
#        if self.headless:
#            self.driver.get(self.url)
#        try:
#            WebDriverWait(self.driver, 5).until(
#                    EC.element_to_be_clickable((By.ID, self.SYNC_SCREEN_ID)))
#        except TimeoutException as e:
#            logging.error('Page did not load properly. Wrong MINIWOB_BASE_URL?')
#            raise e
import json
import logging
import os
from Queue import Queue
import sys
import time
import traceback
import urlparse
from threading import Thread, Event

import numpy as np

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from time import sleep


options = webdriver.ChromeOptions()
options.add_argument('app=file:///Users/Evan/Documents/code/SlimeJavascript/SlimeVolleyballLegacy.html')
#options.add_argument('window-size={},{}'
#        .format(self.window_width, self.window_height))
#options.add_argument('window-position={},{}'
#        .format(9000, 30 + self.index * (self.window_height + 30)))
driver = webdriver.Chrome(chrome_options=options)
driver.implicitly_wait(5)
driver.execute_script('start(false);')
sleep(5)

for _ in xrange(10):
    chain = ActionChains(driver)
    chain.send_keys("wd")
    chain.perform()
    sleep(0.5)

#try:
#    WebDriverWait(self.driver, 5).until(
#            EC.element_to_be_clickable((By.ID, self.SYNC_SCREEN_ID)))
#except TimeoutException as e:
#    logging.error('Page did not load properly. Wrong MINIWOB_BASE_URL?')
#    raise e
