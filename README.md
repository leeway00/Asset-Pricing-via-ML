# Empirical asset pricing via Machine Learning in the Korean market

- Cleaning up and moving the code into .py files

## Methodology
- This project is a replication of **Gu, Kelly, and Xiu, "Empirical Asset Pricing via Machine Learning." Review of Financial Studies, 2020** using data from thr Korean stock market, both KOSPI and KOSDAQ.
- I expanded the neural net models suggested in the paper into models with deeper structure, but the number of factors I gathered here is less than the paper, possibly incurring smaller $R^2$ and more volatile results from the paper's result.

## File explanation
1. ~~Marketdata_crawler~~: Currently dismissed the crawler
2. The factors that I used initially are"
    Beta, SMB, HML, Market portfolio, Moving Average, Momentum, PER
3. ML_pricing: machine learning pricing models. OLS, ElasticNet, PCR, PLS, RandomForest, GBR
4. NN_pricing: Neural net settings of pricing models
5. NN_pricing_changed_setting: I tested several settings of neural nets by changing the optimizers and training methods
6. FF3 test: statistics related to the pricing models, also generate Decile portfolios.

### Additional/Extended Variables
**Additional data revision for this repo**
The data period for training/validation is different in the revised code, which makes the prediction result different from the previous result pdf file.
- index = 'date', 'ticker'
- Target: 'target'
- Variables:
  - Market return based: 'market_return', 'excess_market_return'
  - Fama French 3 factor: 'ff3_bin_return', 'smb', 'hml', 
  - CAPM: 'const', 'beta', 'ido_vol', 'beta_seq', 
  - Fundamental: 'EPR', 'BPR', 'div_ret','div'
  - ETC:
    - 'size_rnk', 'share_turnover', 'share_turnover_rnk', 'std12', 'cross_rnk', 'time_rnk',
    - Momentum:'mom1', 'mom2', 'mom3', 'mom4', 'mom5', 'mom6', 'mom7', 'mom8', 'mom9', 'mom10', 'mom11', 'mom12', 
    - Support line(based on price): 'support_low', 'support_high'
  - Macro
    - Korean Treasury: 'tb3y', 'tb5y', 'tb10y', 'cb3y'
    - usd/krw: 'change_usd_krw_monthly', 'lo_usd_krw_monthly', 'ho_usd_krw_monthly', 'co_usd_krw_monthly', 'change_usd_krw_daily', 'lo_usd_krw_daily', 'ho_usd_krw_daily', 'co_usd_krw_daily',
    - WTI: 'change_wti', 'lo_wti', 'ho_wti', 'co_wti', 
    - Market Portfolio: 'change_nasdaq', 'lo_nasdaq', 'ho_nasdaq', 'co_nasdaq', 'close_sp500', 'change_sp500', 'lo_sp500', 'ho_sp500', 'co_sp500',
    - US Treasury: 'close_bond_10y', 'close_bond_2y', 'close_bond_1m', 'close_bond_1y', 
    - VIX:'close_vix', 'change_vix', 'lo_vix', 'ho_vix', 'co_vix',
  - Log:'log_mom1', 'log_mom2', 'log_mom3', 'log_mom4', 'log_EPR', 'log_share_turnover', 'log_mom6', 'log_mom5', 'log_mom12', 'log_mom11', 'log_mom10', 'log_mom8', 'log_mom9', 'log_mom7', 'log_std12', 'log_BPR', 'log_change_wti', 'log_ff3_bin_return', 'log_ho_wti', 'log_ho_usd_krw_monthly', 'log_smb', 'log_co_sp500', 'log_change_usd_krw_daily', 'log_ho_vix', 'log_change_vix', 'log_change_usd_krw_monthly', 'log_ho_usd_krw_daily', 'log_ho_sp500', 'log_close_vix', 'log_ho_nasdaq', 'log_co_usd_krw_daily', 'log_close_bond_1m', 'log_change_sp500', 'log_co_nasdaq', 'log_close_bond_1y', 'log_close_bond_2y', 'log_ido_vol', 'log_change_nasdaq', 'log_close_sp500', 'log_lo_usd_krw_daily', 'log_lo_nasdaq', 'log_lo_sp500', 'log_lo_usd_krw_monthly', 'log_beta_seq', 'log_beta', 'log_hml', 'log_co_usd_krw_monthly', 'log_lo_wti', 
  - Categorical: 'vix_cat_mid', 'vix_cat_high'

## Limitations of the research
- Data availability. There is a survivorship bias in the data since the only data available through Korea Exchange is for the securites that are currently traded in the market.
- Lack of factors and data that almost 90% of the data used in the Gu's paper was not available within my reachouts.

## Revision
- 2022.05.23:
  - revised the code for ML_pricing
  - make ML_pricing as .py file
