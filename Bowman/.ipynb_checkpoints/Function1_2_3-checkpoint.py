# %%
#Function 1 of ChemLibre Texts reading program, takes in a url, path, and browser type and returns the html
#Path location should be in the format ex. C:/Users/bowri/Anaconda3/chromedriver
#If using Firefox, or not Chrome, simply enter "" for path location, requires having downloaded chromedriver first
#See formatting below

#Stuff to do:
    #1) throw more errors  - check, still work on the try/except for selenium being present
    #2) getting rid of import functions - check
    #3) add docstrings to let user know which types of data are allowed - check
    #4) add default settings, eg. output = none; have output in, maybe more
    #5) document better
    
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import pandas as pd
from lxml import html
import selenium
from bs4 import BeautifulSoup as BS
import random
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException as SERE

def selenium_html_collector(url, browser, path_to_driver, webdriver):
    """"This function takes in three strings: 1) a url, 2) a browser,
        and 3) a path to your local chromedriver location, which is only 
        need if you are using Chrome. It takes in 4) a webdriver module
        from Selenium as well. It returns an html of the given
        url and opens the browser to that site as well""" 
    
    if browser == "Firefox":
        #try:
        drive = webdriver.Firefox()
        #except: 
        #    print("Did you import the webdriver module from Selenium?")   
        
    elif browser == "Chrome":
        #try:
        drive = webdriver.Chrome(executable_path= (path_to_driver))
        #except: 
        #    print("Did you import the webdriver module from Selenium?")

        
    elif browser != "Chrome" or "Firefox":
        print("this is the weird browser:", browser)
        raise Exception("Sorry, the function only utilizes Firefox and Chrome currently")

        
    drive.get(url)
    return drive



def book_finder(url, driver):               #SHOULD GET RID OF INITIALIZED LIST EVERY TIME PROGRAM RUN, ADD AS ARGUMENT
    book_urls = []
    urls = []
    driver.get(url)
    driver.implicitly_wait(random.randint(1,10))
    #mt-sortable-listing-link mt-edit-section internal is the class name that gets all genres
    #can do something recursive, where if <h1 .text contains "Book:" stop
    

    sections = driver.find_elements_by_class_name("mt-sortable-listing-link mt-edit-section internal")
    print(type(sections))
    print(sections)
    #if h1.text does not contain "Book:"
    header = str(driver.find_element_by_xpath("//*[@id='title']").text)
    print(header)
        #for section in sections:
        #    book_finder(section, driver)
    print()
    
    for section in sections:
        urls.append(str(section.get_attribute("href").text))

                    
    print(urls)
    #else:
        #for section in sections:
            #book_url.append = href value(link)
        
    #return book_urls





# %%
#Function 3 of ChemLibreTexts reading program, takes in two lists: 1) chapter titles and 2) chapter 
#contents and 3) a filename, and exports them to a JSON file with the given filename

#Creates a dictionary with the two lists, and writes and opens a json file

#add additional arguments for default settings, eg output_state boolean, for printing vs writing
def chapter_exporter(chapter_titles, chapter_contents, filename, export = True):
    """"This function takes in three variables, and has one default variable. The first two
    variables must be lists, which ultimately get combined into a dictionary. The third var
    is the string filename of your choice, and the final variable determines whether or not
    the program will print or export the dictionary to a json. By default it is set to true"""
    
    if isinstance(chapter_titles, list) and isinstance(chapter_contents, list)  == True:
       
        titles_and_contents = dict(zip(chapter_titles, chapter_contents))

        if export == True:
            with open(filename, "w") as outfile:
                json.dump(titles_and_contents, outfile)
        else:
            print(titles_and_contents)
    else:
         raise Exception("Variables passed in must be lists")

# %%
#import json
#titles_list = ["chapter 1", "chapter 2", "chapter 3"]
#chap_list = ["this is chapter 1", "this is chapter 2", "this is chapter 3"]
#title = "chapter 1"

#chapter_exporter(titles_list, chap_list, "test_chapter_writing", False)

# %%
def chapter_text_parser(driver):

    driver.implicitly_wait(random.randint(1,100))
    chapter_title = driver.find_element(By.XPATH, '//*[@id="title"]').text.strip()
    subchap_link_title_container = driver.find_elements(By.CLASS_NAME, 'mt-listing-detailed-title')
    subchap_titles = [title.text.strip() for title in subchap_link_title_container ]
    subchap_links = [link.find_element_by_tag_name('a').get_attribute('href') for link in subchap_link_title_container]
    print('Name of chapter', chapter_title, '\n', 'Number of subchapter is', len(subchap_links))
    subchap_overview_container = driver.find_elements(By.CLASS_NAME, 'mt-listing-detailed-overview')
    subchap_overviews = [overview.text.strip() for overview in subchap_overview_container]

    subchap_contents=[]
    data = {}
    for chap_link in subchap_links:
        driver.get(chap_link)
        driver.page_source
        chap_text_container = driver.find_elements(By.CLASS_NAME,'mt-content-container') 
        for subchap in chap_text_container:
            subchap_contents.append(subchap.text.strip())     
    data = {'chap-title':subchap_titles, 'chap-overview': subchap_overviews, 'chap-content':subchap_contents}
    return data

# %%
def new_exporter(dictionary, filename, driver, printout = False):
    
    if printout == False:
        with open(filename, "w") as outfile:
            json.dump(dictionary, outfile)
    
    else:
         print(dictionary)   
    return driver.close()
    