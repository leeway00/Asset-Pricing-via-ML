# Empirical asset pricing via Machine Learning in the Korean market
Look at **resaerch-result.pdf** for the result value of the jupyter notebooks.

- If you have any idea to further this repository, please leave me any issue :)
- Cautious: the codes are now very congested with insufficient docstrings
    - Currently cleaning up and moving the code into .py files

## Methodology
- This project is replication of **Gu, Kelly, and Xiu, "Empirical Asset Pricing via Machine Learning." Review of Financial Studies, 2020** using data from thr Korean stock market, both KOSPI and KOSDAQ.
- I expanded neural net models suggested in the paper into models with deeper structure, but the factors I used here are less than the paper.


## File explanation
1. Marketdata_crawler: crawls monthly price data from Korea Exchange(한국거래소). It requires a list of tickers, which also can be gathered by crawling Korea Exchange. I didn't uploaded the ticker csv file I used since the tickers will be changed in future. Also, it does not contains the tickers that temporarily ceased to trade.
2. Factor_generatig: generating factors based on the price data. These are the concepts of factors that I used.
    1. Beta
    2. SMB
    3. HML
    4. Market portfolio
    5. Moving Average
    6. Momentum
    7. PER
3. ML_pricing: machine learning pricing models. OLS, ElasticNet, PCR, PLS, RandomForest, GBR
4. NN_pricing: Neural net settings of pricing models
5. NN_pricing_changed_setting: I tested several settings of neural nets by changing the optimizers and training methods
  - Basically, the results heavily depend on optimizers such that logicalness of using machine learning on pricing is questionnable.
6. FF3 test: statistics related to the pricing models, also generate Decile portfolios.
 


## Limitation of the research
- Data availability. There is a survivorship bias in the data since the only data available through Korea Exchange is for the securites that are currently traded in the market.
- Lack of factors.
