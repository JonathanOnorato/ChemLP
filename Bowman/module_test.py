{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'This program so far imports a variant of the functions we all made as BowFunks. It then takes a given start url, in this case a Chapter from\\nthe Schaller textbook, then goes through each subchapter(ie 1.1 etc) and gets all the content. Right now it just runs through one chapter. \\nTrying to broaden that.'"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from selenium import webdriver\n",
    "from selenium.webdriver.common.keys import Keys\n",
    "import json\n",
    "import pandas as pd\n",
    "from lxml import html\n",
    "import selenium\n",
    "from bs4 import BeautifulSoup as BS\n",
    "import random\n",
    "import time\n",
    "from selenium.webdriver.common.by import By\n",
    "from selenium.webdriver.support import expected_conditions as EC\n",
    "from selenium.common.exceptions import StaleElementReferenceException as SERE\n",
    "\n",
    "import Function1_2_3 as BowFunks  #this is locally on my computer, so just need to figure out a wait to generalize an import\n",
    "\n",
    "#Pseudocode her:\n",
    "#first ask \"user\" for an initial url, basically the one thing we need to run the program, currently at the chapter level\n",
    "    #kind of need to \"look under the hood\" here, can I just run Rukiya/Louisa's program at the Book interface to make it keep going thru\n",
    "    #chapters?\n",
    "\n",
    "#Questions:\n",
    "#determine how many json's we're exporting?\n",
    "#Where do the other import things need to be, in the .py module that we're importing? or in this master function\n",
    "#is there a benefit/loss to using Jupyter Notebook to edit a .py file\n",
    "#how to generalize an import, ie BowFunks is Function1_2_3.py locally for me\n",
    "#what needs to be changed in parser function, such that it works at a broader level\n",
    "\n",
    "#Basically summary:\n",
    "\"\"\"This program so far imports a variant of the functions we all made as BowFunks. It then takes a given start url, in this case a Chapter from\n",
    "the Schaller textbook, then goes through each subchapter(ie 1.1 etc) and gets all the content. Right now it just runs through one chapter. \n",
    "Trying to broaden that.\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'list'>\n",
      "Bookshelves\n",
      "\n",
      "[]\n"
     ]
    }
   ],
   "source": [
    "url = \"https://chem.libretexts.org/Bookshelves\"\n",
    "\n",
    "#returns the driver(not the html, which confused me), of the given url\n",
    "#html_start = BowFunks.selenium_html_collector(url, \"Chrome\", \"C:/Users/bowri/Anaconda3/chromedriver\", webdriver)\n",
    "\n",
    "\n",
    "#this is storing a dictionary of all of chapter 1 of 1 book, need to start going through all books from this level\n",
    "#all_dict = BowFunks.chapter_text_parser(html_start)\n",
    "\n",
    "#exports to single json of a chapter, or prints it out if you enter True into final default variable\n",
    "#BowFunks.new_exporter(all_dict, \"test_exe\", html_start, True)\n",
    "driver = webdriver.Chrome()\n",
    "BowFunks.book_finder(url, driver)\n",
    "\n",
    "#//*[@id=\"title\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
