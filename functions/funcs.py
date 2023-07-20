import pandas as pd
import streamlit as st

def r_score(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
    
def fm_score(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def segmenting_customers(x):
    if x >= 9:
        return 'Top'
    elif x >= 5 and x < 9:
        return 'Middle'
    else:
        return 'Low'