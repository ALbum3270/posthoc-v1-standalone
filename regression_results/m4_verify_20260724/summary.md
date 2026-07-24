# GraphRAG Regression Summary

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Sources | Rounds | Tokens | Chat cost | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ftx_2022 | 100% | 100% | 100% | 91% | 17 | 17 | 36,291 | $0.0237 | 417.7s |
| crowdstrike_2024 | 94% | 100% | 100% | 97% | 15 | 19 | 39,323 | $0.0233 | 360.3s |
| svb_2023 | 100% | 100% | 100% | 94% | 14 | 17 | 36,793 | $0.0231 | 361.5s |

\* Slot relevance is model-judged and is not used for graph writes or stopping.

## ftx_2022

- Stop: `coverage_reached`; facts 96; episodes 34; duplicate queries 0
- Grounding rejections: 8%; dated facts: 25%; cross-corroborated slots: 100%
- ✅ `ftx_bankruptcy_2022` (present) — FTX bankruptcy is anchored to 2022
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
- ✅ `ftx_sbf` (present) — Sam Bankman-Fried is identified
  - `Alameda Research, a trading firm affiliated with FTX and owned by FTX chief executive Sam Bankman-Fried`
  - `Sam Bankman-Fried and the Collapse of FTX`
- ✅ `ftx_alameda` (present) — Alameda Research is connected to the event
  - `Alameda Research, a trading firm affiliated with FTX and owned by FTX chief executive Sam Bankman-Fried`
  - `Sam Bankman-Fried cofounded Alameda Research, a cryptocurrency trading firm, in 2017.`
- ✅ `ftx_bahamas` (present) — The Bahamas appears as FTX's operating jurisdiction
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
- ✅ `ftx_no_2023_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2023
- ✅ `ftx_no_2026_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2026

## crowdstrike_2024

- Stop: `all_slots_exhausted`; facts 78; episodes 30; duplicate queries 0
- Grounding rejections: 13%; dated facts: 17%; cross-corroborated slots: 88%
- ✅ `crowdstrike_date` (present) — The outage is anchored to July 19, 2024
  - `Date | 19 July 2024; 2 years ago (2024-07-19)`
  - `On Friday, July 19, 2024, nearly 8.5 million Microsoft devices were affected by a faulty system update, causing a major outage of businesses and services worldwide.`
- ✅ `crowdstrike_falcon` (present) — CrowdStrike Falcon is identified as the affected product
  - `CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software, triggering catastrophic`
- ✅ `crowdstrike_windows` (present) — Windows systems are identified as affected
  - `Computers running macOS and Linux were unaffected, as the problematic content file was only for Windows`
  - `CrowdStrike is actively working with customers impacted by a defect found in a single content update for Windows hosts`
- ✅ `crowdstrike_update` (present) — A faulty content/software update is identified as the trigger
  - `Multiple blue screens of death caused by a faulty software update on baggage carousels at LaGuardia Airport, New York City`
  - `CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software, triggering catastrophic`
- ✅ `crowdstrike_no_2025_event` (absent) — The 2024 outage is not dated to 2025
- ✅ `crowdstrike_no_2026_event` (absent) — The 2024 outage is not dated to 2026

## svb_2023

- Stop: `coverage_reached`; facts 85; episodes 33; duplicate queries 0
- Grounding rejections: 4%; dated facts: 26%; cross-corroborated slots: 94%
- ✅ `svb_closure_date` (present) — SVB's closure is anchored to March 10, 2023
  - `On March 10, 2023, Silicon Valley Bank (SVB) failed after a bank run`
- ✅ `svb_fdic` (present) — The FDIC is identified as an intervening authority
  - `Two days after the failure, the FDIC received exceptional authority from the Treasury`
  - `placed it under the receivership of the Federal Deposit Insurance Corporation (FDIC)`
- ✅ `svb_bank_run` (present) — Depositor withdrawals or a bank run appear in the mechanism
  - `On March 10, 2023, Silicon Valley Bank (SVB) failed after a bank run, marking the third-largest bank failure in United States history and the largest since the 2008 financial crisis.`
  - `SVB faced a bank run as depositors rushed to withdraw their funds amid rising inflation and deteriorating financial conditions in the tech sector`
- ✅ `svb_rates_bonds` (present) — Interest rates and securities losses appear in the explanation
  - `SVB had heavily invested deposits into long-term treasury bonds, which lost value as the Federal Reserve raised interest rates. Consequently, the bank was unable to meet withdrawal demands, leading to significant losses when it had to sell these bonds at unfavorable prices.`
  - `SVB had heavily invested deposits into long-term treasury bonds, which lost value as the Federal Reserve raised interest rates`
- ✅ `svb_no_2022_closure` (absent) — SVB's closure is not dated to 2022
- ✅ `svb_no_2024_closure` (absent) — SVB's closure is not dated to 2024
