{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'chapter 1': 'this is chapter 1', 'chapter 2': 'this is chapter 2', 'chapter 3': 'this is chapter 3'}\n"
     ]
    }
   ],
   "source": [
    "from selenium import webdriver\n",
    "from selenium.webdriver.common.keys import Keys\n",
    "import json\n",
    "\n",
    "import Function1_3 as BowFunks\n",
    "\n",
    "BowFunks.selenium_html_collector(\"https://www.wikipedia.org/\", \"Chrome\", \"C:/Users/bowri/Anaconda3/chromedriver\", webdriver)\n",
    "\n",
    "titles_list = [\"chapter 1\", \"chapter 2\", \"chapter 3\"]\n",
    "chap_list = [\"this is chapter 1\", \"this is chapter 2\", \"this is chapter 3\"]\n",
    "title = \"chapter 1\"\n",
    "\n",
    "BowFunks.chapter_exporter(titles_list, chap_list, \"test_chapter_writing\", False)\n"
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
