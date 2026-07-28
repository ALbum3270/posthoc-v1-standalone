# GraphRAG Regression Summary

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Claim corroboration | Critical support | Sources | Rounds | Tokens | Chat cost | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| svb_2023 | 100% | 100% | 100% | 100% | 22% | 86% | 9 | 21 | 46,756 | $0.0283 | 324.6s |

\* Slot relevance is model-judged and is not used for graph writes or stopping.

## svb_2023

- Stop: `coverage_reached`; facts 27; episodes 22; duplicate queries 0
- Grounding rejections: 18%; dated facts: 52%; multi-source slots: 31%; claim corroboration: 22%
- Pre-write relevance: 33 accepted, 0 uncertain, 13 rejected; search results rejected: 6; targeted support rounds: 5
- ✅ `svb_closure_date` (present) — SVB's closure is anchored to March 10, 2023
  - `On Friday, March 10, 2023, Silicon Valley Bank, Santa Clara, CA was closed by the California Department of Financial Protection & Innovation and the Federal Deposit Insurance Corporation (FDIC) was named Receiver.`
  - `On Friday, March 10, 2023, Silicon Valley Bank, Santa Clara, CA was closed by the California Department of Financial Protection & Innovation and the Federal Deposit Insurance Corporation (FDIC) was named Receiver.`
- ✅ `svb_fdic` (present) — The FDIC is identified as an intervening authority
  - `Press Release, FDIC,Joint Statement by the Department of the Treasury, Federal Reserve, and FDIC, Fed. Deposit
Ins. Corp. (Mar. 12, 2023)`
  - `Prior to establishing the SVB bridge bank, the FDIC briefly established a Depository Institution National Bank
(DINB) with the expectation of liquidating the bank through a deposit payoff. See Press Release, FDIC, FDIC Creates a
Deposit Insurance National Bank of Santa Clara to Protect Insured Depositors of Silicon Valley Bank, Santa Clara,
California, Fed. Deposit Ins. Corp. (Mar. 10, 2023)`
- ✅ `svb_bank_run` (present) — Depositor withdrawals or a bank run appear in the mechanism
  - `SVB faced a bank run as depositors rushed to withdraw their funds amid rising inflation and deteriorating financial conditions in the tech sector, where many of its clients operated.`
  - `SVB had heavily invested deposits into long-term treasury bonds, which lost value as the Federal Reserve raised interest rates. Consequently, the bank was unable to meet withdrawal demands, leading to significant losses when it had to sell these bonds at unfavorable prices.`
- ✅ `svb_rates_bonds` (present) — Interest rates and securities losses appear in the explanation
  - `SVB had heavily invested deposits into long-term treasury bonds, which lost value as the Federal Reserve raised interest rates. Consequently, the bank was unable to meet withdrawal demands, leading to significant losses when it had to sell these bonds at unfavorable prices.`
- ✅ `svb_no_2022_closure` (absent) — SVB's closure is not dated to 2022
- ✅ `svb_no_2024_closure` (absent) — SVB's closure is not dated to 2024
