#!/usr/bin/env python
# coding: utf-8



from pdfminer.high_level import extract_text
import json
import re as re
import string




def extract_text_from_pdf(pdf_path):
    global cleanest_text
    text = extract_text(pdf_path)
    pre_clean_text = text.lower().strip().replace('\ufb01', 'fi').replace('\ufb02', 'fl').replace('\u00e1', 'a').replace('\u00f6', 'o').replace('\u2014', '-').replace('\f', ' ')
    clean_text = pre_clean_text.replace('a\u02da', 'angstroms').replace('\u20aca', 'a').replace('\u00fc', 'u').replace('\u00e9', 'e').replace('\u2013', '-').replace('\n', ' ')
    cleaner_text = re.sub('^\\\\u[\d\D]{4}|(cid:\d)|(cid:\d\d)|^\\\\xa0', '', clean_text)
    cleanest_text = cleaner_text.replace('\xa0', ' ')
    if cleanest_text:
        return cleanest_text



def json_exporter_adapted(filename):
    if cleanest_text:
        print(cleanest_text)
        with open(filename, "w") as outfile:
            json.dump(cleanest_text, outfile).encode('ascii')
    return 




