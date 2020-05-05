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

        
    html = drive.get(url)
    return html



# %%
#Test Runs
#import selenium
#from selenium import webdriver
#from selenium.webdriver.common.keys import Keys

#selenium_html_collector("https://www.wikipedia.org/", "Chrome", "C:/Users/bowri/Anaconda3/chromedriver")
#selenium_html_collector("https://www.wikipedia.org/", "Firefox", "")


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


# %%
