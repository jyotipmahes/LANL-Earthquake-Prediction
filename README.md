# LANL-Earthquake-Prediction
A group project by Anush Kocharyan, Jyoti Prakash Maheswari, Viviana M´arquez, Wenkun Xiao

## Objective
This project was taken as a part of MSDS697 course. The main objective of the project was to create a distributed database and spark machine learning pipeline on a big dataset. We selected the Earthquake dataset from [LANL Earthquake Prediction Challenge](https://www.kaggle.com/c/LANL-Earthquake-Prediction) on Kaggle. The dataset consists of ~9gb of raw data collected from a laboratory Earthquake experiment. Our goal was to **extract features** from this dataset, store it on a **distributed and sharded MongoDB server** and use this data to build **Spark ML models using AWS EMR clusters**.

## Steps
**Storing on S3**: We started by storing the whole data on a S3 bucket.<br>
**Feature Extracting using EC2**: The raw data is a time series data. Our objective is to look at a sequence of 150000 data points and predict the time for the next Earthquake from the last time step. We decided to extract a lot of time series features. Since the data is large to fit on a local system, we use **AWS EC2** instance for feature engineering. We used **c5d.4xlarge** instance to created 700K featured datapoints. <br>
**MongoDB**: We store the featured data by creating a MongoDB database. We created a shared and distributed database consisting of 2 shards and 3 nodes per shard. We also created 3 node server and 2 configuration server(a total 0f 11 servers). Please check out [instructions](Deploy_MongoDB_on_AWS_v013.pdf) on how to deploy MongoDB servers on AWS.<br>
**Modelling**: We use AWS EMR clusters at different configuration to build Spark ML models. <br>

## Results
In our experiment, we found that Random Forest was the best performing
algorithm. For that reason, we decided to run cross-validation to further
improve our results. Our lowest RMSE score is 1.40, which is a significant
improvement from our previous score of 2.20 for Random Forest with default
parameters.<br>
**Instance specs**
We ran our experiments in four clusters and obtained the following speed
results:<br>

|Instance/Execution time| Decision Tree| Linear Regression| Random Forest|
|----------|------:|------:|------:|
|m4.2xlarge| 1.68s| 1.10s| 5.13s|
|m4.xlarge| 21.9s| 2.39s| 5.81s|
|m4.large| 57.9s |4.15s |10.5s|
|m3.xlarge|2.93s|1.73s|7.47s|

## Learning
In this project, we learned how to store data using MongoDB on Amazon
Web Services. We did this by creating two shards and three replicas, in that
way, we distributed the data and mirrored the data to obtain fault-tolerance
in our project.
We spent a significant amount of time setting up EMR cluster to run Spark
in Jupyter Notebook. In the process, we learned how to install various dependencies on EMR cluster and configuring various profile values so that we
could start our machine learning experiments.
In addition, we gained experience with SparkSQL and Spark ML on a real life
project. More importantly, we also learned how to be patient and perseverant
when working on real life data science projects.<br>

*Check out the complete report [here](Report.pdf)*
