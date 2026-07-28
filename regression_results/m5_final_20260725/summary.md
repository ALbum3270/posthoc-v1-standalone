# GraphRAG Regression Summary

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Claim corroboration | Critical support | Sources | Rounds | Tokens | Chat cost | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ftx_2022 | 82% | 100% | 100% | 100% | 5% | 20% | 10 | 24 | 45,067 | $0.0261 | 282.1s |
| crowdstrike_2024 | 100% | 100% | 100% | 100% | 20% | 67% | 13 | 23 | 51,537 | $0.0292 | 394.6s |
| svb_2023 | 100% | 83% | 100% | 100% | 27% | 86% | 12 | 23 | 52,132 | $0.0295 | 295.5s |

\* Slot relevance is model-judged and is not used for graph writes or stopping.

## ftx_2022

- Stop: `max_rounds`; facts 22; episodes 15; duplicate queries 0
- Grounding rejections: 14%; dated facts: 36%; multi-source slots: 7%; claim corroboration: 5%
- Pre-write relevance: 23 accepted, 0 uncertain, 21 rejected; search results rejected: 31; targeted support rounds: 0
- ✅ `ftx_bankruptcy_2022` (present) — FTX bankruptcy is anchored to 2022
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
  - `On 16 November 2022, the cryptocurrency brokerage service Genesis suspended withdrawals following FTX declaring bankruptcy, further affecting the industry.`
- ✅ `ftx_sbf` (present) — Sam Bankman-Fried is identified
  - `On 2 November 2022, CoinDesk published an article stating that Alameda Research, a trading firm affiliated with FTX and owned by FTX chief executive Sam Bankman-Fried, held a significant amount of FTX's exchange token, FTT.`
- ✅ `ftx_alameda` (present) — Alameda Research is connected to the event
  - `In November 2022 CoinDesk also raised concerns stating that FTX's partner firm Alameda Research held a significant portion of its assets in FTX's native token (FTT).`
  - `FTX collapsed due to severe mismanagement, commingling its funds with its sister firm Alameda Research, and a lack of internal controls and risk management.`
- ✅ `ftx_bahamas` (present) — The Bahamas appears as FTX's operating jurisdiction
  - `On November 11, 2022, cryptocurrency exchange FTX Trading Ltd., incorporated in Antigua and Barbuda and headquartered in The Bahamas, also called FTX.com, filed a petition for relief under Chapter 11 of the Bankruptcy Code in the Bankruptcy Court for the District of Delaware (Case No. 22-11068 JTD)`
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
- ✅ `ftx_no_2023_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2023
- ✅ `ftx_no_2026_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2026

## crowdstrike_2024

- Stop: `coverage_reached`; facts 20; episodes 19; duplicate queries 0
- Grounding rejections: 19%; dated facts: 40%; multi-source slots: 27%; claim corroboration: 20%
- Pre-write relevance: 24 accepted, 0 uncertain, 22 rejected; search results rejected: 15; targeted support rounds: 7
- ✅ `crowdstrike_date` (present) — The outage is anchored to July 19, 2024
  - `| Date | 19 July 2024; 2 years ago (2024-07-19) |`
  - `July 19, 2024, 05:27 UTC: CrowdStrike identified the issue and reverted the changes, but by then, many systems had already been affected.`
- ✅ `crowdstrike_falcon` (present) — CrowdStrike Falcon is identified as the affected product
  - `Information on how the update to the CrowdStrike Falcon sensor configuration file, Channel File 291, caused the logic error that led to the outage.`
  - `On July 19, 2024, CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software`
- ✅ `crowdstrike_windows` (present) — Windows systems are identified as affected
  - `On July 19, 2024, a defective update from CrowdStrike, an American cybersecurity company, triggered a major IT outage for over 8.5 million devices running the Microsoft system.`
  - `In July 2024, millions of Windows users were locked out of their systems due to a flaw in a CrowdStrike update.`
- ✅ `crowdstrike_update` (present) — A faulty content/software update is identified as the trigger
  - `On July 19, 2024, a defective update from CrowdStrike, an American cybersecurity company, triggered a major IT outage for over 8.5 million devices running the Microsoft system.`
  - `On July 19, 2024, CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software`
- ✅ `crowdstrike_no_2025_event` (absent) — The 2024 outage is not dated to 2025
- ✅ `crowdstrike_no_2026_event` (absent) — The 2024 outage is not dated to 2026

## svb_2023

- Stop: `coverage_reached`; facts 22; episodes 22; duplicate queries 0
- Grounding rejections: 10%; dated facts: 55%; multi-source slots: 31%; claim corroboration: 27%
- Pre-write relevance: 28 accepted, 0 uncertain, 19 rejected; search results rejected: 14; targeted support rounds: 5
- ✅ `svb_closure_date` (present) — SVB's closure is anchored to March 10, 2023
  - `On Friday, March 10, 2023, Silicon Valley Bank, Santa Clara, CA was closed by the California Department of Financial Protection & Innovation and the Federal Deposit Insurance Corporation (FDIC) was named Receiver.`
  - `On Friday, March 10, 2023, Silicon Valley Bank, Santa Clara, CA was closed by the California Department of Financial Protection & Innovation and the Federal Deposit Insurance Corporation (FDIC) was named Receiver.`
- ❌ `svb_fdic` (present) — The FDIC is identified as an intervening authority
- ✅ `svb_bank_run` (present) — Depositor withdrawals or a bank run appear in the mechanism
  - `When economic problems hit the tech sector, many bank customers withdrew money as venture capital started drying up.`
  - `SVB had heavily invested deposits into long-term treasury bonds, which lost value as the Federal Reserve raised interest rates. Consequently, the bank was unable to meet withdrawal demands, leading to significant losses when it had to sell these bonds at unfavorable prices.`
- ✅ `svb_rates_bonds` (present) — Interest rates and securities losses appear in the explanation
  - `When the Federal Reserve hiked interest rates in 2022 to counter inflation, SVB’s bond portfolio started to drop.`
  - `SVB had heavily invested deposits into long-term treasury bonds, which lost value as the Federal Reserve raised interest rates. Consequently, the bank was unable to meet withdrawal demands, leading to significant losses when it had to sell these bonds at unfavorable prices.`
- ✅ `svb_no_2022_closure` (absent) — SVB's closure is not dated to 2022
- ✅ `svb_no_2024_closure` (absent) — SVB's closure is not dated to 2024
