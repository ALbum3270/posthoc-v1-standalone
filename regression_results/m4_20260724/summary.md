# GraphRAG Regression Summary

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Sources | Rounds | Tokens | Chat cost | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crowdstrike_2024 | 100% | 100% | 100% | 79% | 11 | 17 | 18,758 | $0.0104 | 214.5s |
| ftx_2022 | 100% | 100% | 100% | 100% | 7 | 18 | 21,763 | $0.0130 | 211.8s |
| svb_2023 | 100% | 67% | 100% | 91% | 6 | 17 | 18,961 | $0.0116 | 225.5s |

\* Slot relevance is model-judged and is not used for graph writes or stopping.

## crowdstrike_2024

- Stop: `coverage_reached`; facts 29; episodes 17; duplicate queries 0
- Grounding rejections: 3%; dated facts: 14%; cross-corroborated slots: 0%
- ✅ `crowdstrike_date` (present) — The outage is anchored to July 19, 2024
  - `"Faulty CrowdStrike update causes major global IT outage, taking out banks, airlines and businesses globally". TechCrunch. Archived from the original on 19 July 2024. Retrieved 19 July 2024.`
  - `"Massive outage hits companies around the world". news.com.au. 19 July 2024. Retrieved 19 July 2024.`
- ✅ `crowdstrike_falcon` (present) — CrowdStrike Falcon is identified as the affected product
  - `a flawed update to CrowdStrike’s Falcon sensor, a cybersecurity tool widely used to protect devices like computer workstations and servers, resulted in a global IT outage.`
  - `CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software`
- ✅ `crowdstrike_windows` (present) — Windows systems are identified as affected
  - `Microsoft released a recovery tool that uses a USB drive to boot and repair affected systems.
Microsoft also published a blog post that provides links to various remediation solutions and outlines their actions in response to the outage, which include working with CrowdStrike to expedite restoring services to disrupted systems.`
  - `In the blog post, Microsoft estimates the outage affected 8.5 million Windows devices.`
- ✅ `crowdstrike_update` (present) — A faulty content/software update is identified as the trigger
  - `CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software`
  - `a faulty software update from cybersecurity firm CrowdStrike caused what is broadly considered one of the most widespread IT outages in history`
- ✅ `crowdstrike_no_2025_event` (absent) — The 2024 outage is not dated to 2025
- ✅ `crowdstrike_no_2026_event` (absent) — The 2024 outage is not dated to 2026

## ftx_2022

- Stop: `coverage_reached`; facts 43; episodes 17; duplicate queries 0
- Grounding rejections: 10%; dated facts: 26%; cross-corroborated slots: 0%
- ✅ `ftx_bankruptcy_2022` (present) — FTX bankruptcy is anchored to 2022
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
- ✅ `ftx_sbf` (present) — Sam Bankman-Fried is identified
  - `Explainer: What happens next in Sam Bankman-Fried's fraud case`
  - `Sam Bankman-Fried Posts Weird Cryptic Tweets After Wealth Wipeout`
- ✅ `ftx_alameda` (present) — Alameda Research is connected to the event
  - `Sam Bankman-Fried says Alameda Research to wind down trading, FTX attempting to raise capital`
  - `On 2 November 2022, CoinDesk published an article stating that Alameda Research, a trading firm affiliated with FTX and owned by FTX chief executive Sam Bankman-Fried, held a significant amount of FTX's exchange token, FTT.`
- ✅ `ftx_bahamas` (present) — The Bahamas appears as FTX's operating jurisdiction
  - `Nassau, New Providence , The Bahamas |`
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
- ✅ `ftx_no_2023_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2023
- ✅ `ftx_no_2026_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2026

## svb_2023

- Stop: `coverage_reached`; facts 33; episodes 17; duplicate queries 0
- Grounding rejections: 6%; dated facts: 36%; cross-corroborated slots: 0%
- ❌ `svb_closure_date` (present) — SVB's closure is anchored to March 10, 2023
- ✅ `svb_fdic` (present) — The FDIC is identified as an intervening authority
  - `The FDIC in turn has created the Deposit Insurance National Bank of Santa Clara, which now holds the insured deposits from SVB`
  - `the California Department of Financial Protection and Innovation closed SVB and named the FDIC as the receiver`
- ✅ `svb_bank_run` (present) — Depositor withdrawals or a bank run appear in the mechanism
  - `In two days, SVB went from functioning to insolvent when depositors rushed to SVB to withdraw their funds.`
- ❌ `svb_rates_bonds` (present) — Interest rates and securities losses appear in the explanation
- ✅ `svb_no_2022_closure` (absent) — SVB's closure is not dated to 2022
- ✅ `svb_no_2024_closure` (absent) — SVB's closure is not dated to 2024
