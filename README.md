# Predicting the effect of nursing home quality measures on long-stay emergency department outpatient visits 
author: Erica Pinto 

## Repository Contents
This repository contains the code required to evaluate the impact of select nursing home quality measures on the long-stay emergency department outpatient visit rate within the United States utilizing a Stepwise Linear Regression Model and a XGBoosted Random Forest Regression Model. 

# Table of Contents
1. [Abstract](#Abstract)
2. [Testing](#Something_else)


## Abstract 
### Context
As of 2016, approximately 11% of the United States 85 years and older population live within nursing homes on a long-term basis, 69% of which have at least one disability that affects their quality of life – hearing, vision, cognitive, ambulation (Roberts et al., 2018). Frequent unplanned avoidable and unavoidable transfers from the nursing home to the emergency department can further negatively impact residents’ health status, can hinder care due to gaps in communication during transition, and can be costly for Medicaid programs (Moccia & Keyes, 2021; Walsh et al., 2010). Utilizing machine learning techniques, the relationship between operational, clinical and safety nursing home quality attributes and rate of emergency department visits for long-stay patients from the facility will be explored.
### Problem Statement
Research questions for this study are as follows:
Which of the following attributes affect outpatient emergency department visits from nursing homes for long-stay patients?
• Nursing home facility ownership type (for-profit versus government-owned)
• Bed numbers and bed occupancy rate
• Adjusted staffing hours per day across all nursing disciplines and turnover rate
• Reported incidents: facility-reported incidents, substantiated complaints, infection control citations
• Long-stay patient quality rating
• Inspection survey outcomes on 2 domains: health deficiencies and fire safety
• MDS 3.0 clinical measures for long-stay patients
• COVID-19 incidence and death rates
Through utilization of machine learning techniques, is it possible to predict which of the above features impact return to outpatient emergency department incidence?
### Data
Nursing home demographics and attributes are from:
• The Centers for Medicare & Medicaid Services (CMS) Provider Information dataset
• The CMS MDS Quality Measures dataset
• The CMS Medicare Survey Summary Measures dataset
• The CMS COVID-19 dataset
Aside from facility ownership type, which is a categorical variable, the independent variables are numeric and continuous.
The rate of outpatient emergency department visits is from the CMS Medicare Claims Quality Measures dataset (Measure Code 552). It is a numeric continuous variable that is calculated as a proportion of outpatient emergency department visits per 1000 long-stay resident days.
### Techniques and Tools
Python will be used across the lifecycle of this study including data preprocessing and exploration. Two approaches will be adopted:
1. Random forest regression - will be used to build a model and make predictions. Chosen due to robustness of algorithm for potentially non-linear data and size of data set.
2. Stepwise regression – will be used to build a model in Python. Chosen due to high number of features that may be highly correlated.
If time permits, a neural network model may be implemented as well. Models will use an 80:20 train/test split to train, test and validate the model. In addition, the regression model will use k-fold cross-validation. 
### Evaluation
Model evaluation metrics will be:
• Root Mean Square Error
• Mean Absolute Error
• R2

## Datasets & Requirements 
### Datasets 
Data sets were obtained from the CMS website at the following addresses for the year 2020 (as available): 
- CMS Medicare Claims Quality Measures: https://data.cms.gov/provider-data/dataset/ijh5-nb2v
- CMS MDS Quality Measures: https://data.cms.gov/provider-data/dataset/djen-97ju
- CMS Provider Information: https://data.cms.gov/provider-data/dataset/4pq5-n9py
- CMS Survey Summary: https://data.cms.gov/provider-data/dataset/tbry-pc2d
- CMS COVID-19 Nursing Home Data: https://data.cms.gov/covid-19/covid-19-nursing-home-data

### Requirements 
- Python 3.0 

Required packages are as follows: 
- Pandas
- Numpy 
- OS
- Matplotlib.pyplot
- Seaborn
- Stasmodel.api 
- Scipy.stats
- XGBoost
