### contextual vs standard on fiqa (3 sentences/chunk)

| system | nDCG@10 [95% CI] | recall@100 |
|---|---|---|
| standard, whole doc | 0.519 [0.491, 0.549] | 0.828 |
| standard, chunks (max-pool) | 0.441 [0.413, 0.470] | 0.774 |
| contextual, chunks (max-pool) | 0.454 [0.426, 0.484] | 0.789 |
| contextual chunks, std-model queries (max-pool) | 0.474 [0.445, 0.502] | 0.805 |

contextual (ctx queries) minus standard (chunks), nDCG@10: +0.013 [-0.006, +0.031]
contextual (std queries) minus standard (chunks), nDCG@10: +0.033 [+0.015, +0.049]

Biggest wins for standard chunks (inspect these):
- [-1.00] What is the average cost of a portfolio on a trading site?
- [-1.00] Is is possible to take a mortgage using Bitcoin as collateral?
- [-0.63] How to account for money earned and spent prior to establishing business bank accounts?
- [-0.63] Finance, Social Capital IPOA.U
- [-0.63] Trading with Settled / Unsettled Funds (T+3)

Biggest wins for contextual chunks (inspect these):
- [+1.00] Which colors can one use to fill out a check in the US?
- [+1.00] Can a entrepreneur hire a self-employed business owner?
- [+1.00] Buy/Selling prices at the stock exchange represent someone Selling/Buying at that price?
- [+1.00] How much do brokerages pay exchanges per trade?
- [+1.00] How does revenue shared with someone else go into my tax return in Canada?