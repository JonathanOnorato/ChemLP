{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
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
    "#first ask \"user\" for an initial url, basically the one thing we need to run the program\n",
    "\n",
    "#Questions:\n",
    "#determine how many json's we're exporting?\n",
    "#Where do the other import things need to be, in the .py module that we're importing? or in this master function\n",
    "\n",
    "#Basically summary:\n",
    "\"\"\"This program so far imports a variant of the functions we all made as BowFunks. It then takes a given start url, in this case a Chapter from\n",
    "the Schaller textbook, then goes through each subchapter(ie 1.1 etc) and gets all the content. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Name of chapter 1: Monomers and Polymers \n",
      " Number of subchapter is 9\n"
     ]
    }
   ],
   "source": [
    "url = \"https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Book%3A_Polymer_Chemistry_(Schaller)/1%3A_Monomers_and_Polymers\"\n",
    "html_start = BowFunks.selenium_html_collector(url, \"Chrome\", \"C:/Users/bowri/Anaconda3/chromedriver\", webdriver)\n",
    "\n",
    "\n",
    "#do something where parser takes in a html and driver from html_collector, and returns a dictionary\n",
    "all_dict = BowFunks.chapter_text_parser(html_start)\n",
    "\n",
    "BowFunks.new_exporter(all_dict, \"test_exe\", html_start, False)\n",
    "\n"
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
