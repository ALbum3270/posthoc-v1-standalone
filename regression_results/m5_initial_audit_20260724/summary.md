# GraphRAG Regression Summary

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Claim corroboration | Critical support | Sources | Rounds | Tokens | Chat cost | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ftx_2022 | 100% | 100% | 100% | 100% | 2% | 6% | 8 | 18 | 40,953 | $0.0259 | 331.6s |
| crowdstrike_2024 | 100% | 100% | 100% | 100% | 17% | 67% | 13 | 17 | 43,143 | $0.0253 | 359.0s |
| svb_2023 | 100% | 83% | 100% | 100% | 4% | 14% | 8 | 18 | 45,852 | $0.0289 | 439.4s |

\* Slot relevance is model-judged and is not used for graph writes or stopping.

## ftx_2022

- Stop: `coverage_reached`; facts 45; episodes 18; duplicate queries 0
- Grounding rejections: 20%; dated facts: 29%; multi-source slots: 6%; claim corroboration: 2%
- ✅ `ftx_bankruptcy_2022` (present) — FTX bankruptcy is anchored to 2022
  - `In November 2022, crypto platform FTX Trading and its close affiliates, West Realm Shires Services Inc. and Alameda Research, plus approximately 130 additional affiliated companies, commenced voluntary proceedings under Chapter 11 of the United States Bankruptcy Code.`
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
- ✅ `ftx_sbf` (present) — Sam Bankman-Fried is identified
  - `Alameda Research, a trading firm affiliated with FTX and owned by FTX chief executive Sam Bankman-Fried`
  - `On 2 November 2023, Sam Bankman-Fried was convicted of defrauding customers of FTX and lenders of Alameda Research.`
- ✅ `ftx_alameda` (present) — Alameda Research is connected to the event
  - `Alameda Research, a trading firm affiliated with FTX and owned by FTX chief executive Sam Bankman-Fried`
  - `On 2 November 2023, Sam Bankman-Fried was convicted of defrauding customers of FTX and lenders of Alameda Research.`
- ✅ `ftx_bahamas` (present) — The Bahamas appears as FTX's operating jurisdiction
  - `Nassau, New Providence , The Bahamas |`
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
- ✅ `ftx_no_2023_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2023
- ✅ `ftx_no_2026_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2026

## crowdstrike_2024

- Stop: `coverage_reached`; facts 23; episodes 19; duplicate queries 0
- Grounding rejections: 23%; dated facts: 30%; multi-source slots: 27%; claim corroboration: 17%
- ✅ `crowdstrike_date` (present) — The outage is anchored to July 19, 2024
  - `| Date | 19 July 2024; 2 years ago (2024-07-19) |`
  - `On July 19, 2024, cybersecurity firm CrowdStrike pushed an update to its software that caused some devices running the Windows operating system to crash.`
- ✅ `crowdstrike_falcon` (present) — CrowdStrike Falcon is identified as the affected product
  - `the American cybersecurity company CrowdStrike distributed a faulty update to its Falcon Sensor security software`
- ✅ `crowdstrike_windows` (present) — Windows systems are identified as affected
  - `Computers running macOS and Linux were unaffected, as the problematic content file was only for Windows`
  - `The impact to companies in the Central United States was exacerbated by an unrelated outage with Microsoft Azure the previous day.`
- ✅ `crowdstrike_update` (present) — A faulty content/software update is identified as the trigger
  - `On July 19, 2024, a global IT outage was triggered by a faulty update from CrowdStrike, a leading cybersecurity firm.`
  - `The culprit? A faulty automatic sensor configuration update.`
- ✅ `crowdstrike_no_2025_event` (absent) — The 2024 outage is not dated to 2025
- ✅ `crowdstrike_no_2026_event` (absent) — The 2024 outage is not dated to 2026

## svb_2023

- Stop: `coverage_reached`; facts 27; episodes 17; duplicate queries 0
- Grounding rejections: 6%; dated facts: 56%; multi-source slots: 6%; claim corroboration: 4%
- ✅ `svb_closure_date` (present) — SVB's closure is anchored to March 10, 2023
  - `On Friday, March 10, 2023, the California Department of Financial Protection and Innovation closed Silicon Valley Bank (“SVB”) and appointed the Federal Deposit Insurance Corporation (“FDIC”) as receiver.`
  - `Silicon Valley Bank, Santa Clara, California (“SVB”), was closed on Friday, March 10, 2023 by the California Department of Financial Protection & Innovation`
- ✅ `svb_fdic` (present) — The FDIC is identified as an intervening authority
  - `On March 26, 2023, the FDIC announced that First Citizens BancShares will acquire the commercial banking business of SVB.`
- ✅ `svb_bank_run` (present) — Depositor withdrawals or a bank run appear in the mechanism
  - `On March 10, 2023, Silicon Valley Bank (SVB) failed after a bank run`
  - `On March 10, 2023, Silicon Valley Bank (SVB) failed after a bank run`
- ❌ `svb_rates_bonds` (present) — Interest rates and securities losses appear in the explanation
- ✅ `svb_no_2022_closure` (absent) — SVB's closure is not dated to 2022
- ✅ `svb_no_2024_closure` (absent) — SVB's closure is not dated to 2024
