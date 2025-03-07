# $AAPL stock sentiment analysis

This project uses three different approaches in order to try and classify financial text data into two specific categories: ‘Bullish’ and ‘Bearish’. 

## Table of Contents
- [Motivation](#motivation)
- [Usage](#usage)
- [Results](#results)
- [Questions](#questions)


## Motivation <a name=“motivation”></a>
Being able to predict stock price movement is a paramount goal in the financial world. StockTwits is a social media platform, similar to X (Twitter) designed for sharing thoughts on stocks, crypto and markets between other users. Being able to analyse the sentiment in these discussions would provide very valuable insights into the market sentiment trend which in turn may be correlated to stock market movements. 
 
In this project, I have performed sentiment analysis on any post related to the stock $AAPL, which have been extracted from StockTwits from January 2021. The benefit of using StockTwits is that it has user-initiated sentiment data; the users have the choice to upload a label on their post of either ‘Bullish’ or ‘Bearish’. 

## Usage <a name=“usage”></a>
To use the dataset in this notebook please
1. Use the dl_intro environment
2. Download the [Data](AAPL_2021.csv) in this GitHub repository
3. Upload the dataset file to the Jupiter Notebook environment 
4. Run the code to read the dataset into a Pandas DataFrame

## Results <a name=“results”></a>
The results of the project and more details on the choice of approaches are in a separate file for your reference. The file is in [Results](Results.md).
