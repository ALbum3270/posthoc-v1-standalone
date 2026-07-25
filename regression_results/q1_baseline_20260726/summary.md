# GraphRAG Regression Summary

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Claim corroboration | Critical support | Sources | Rounds | Tokens | Chat cost | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ftx_2022 | 100% | 67% | 100% | 100% | 27% | 86% | 9 | 17 | 36,793 | $0.0204 | 215.3s |
| crowdstrike_2024 | 100% | 100% | 100% | 100% | 32% | 86% | 12 | 14 | 35,720 | $0.0212 | 260.4s |
| svb_2023 | 100% | 83% | 100% | 95% | 27% | 86% | 8 | 16 | 31,434 | $0.0187 | 260.3s |
| turkiye_quake_2023 | 80% | 83% | 100% | 100% | 0% | 0% | 5 | 20 | 24,463 | $0.0141 | 216.6s |

\* Slot relevance is model-judged and is not used for graph writes or stopping.

## ftx_2022

- Stop: `coverage_reached`; facts 22; episodes 16; duplicate queries 0
- Grounding rejections: 22%; dated facts: 27%; multi-source slots: 50%; claim corroboration: 27%
- Pre-write relevance: 28 accepted, 0 uncertain, 0 rejected; search results rejected: 8; targeted support rounds: 7
- ❌ `ftx_bankruptcy_2022` (present) — FTX bankruptcy is anchored to 2022
- ❌ `ftx_sbf` (present) — Sam Bankman-Fried is identified
- ✅ `ftx_alameda` (present) — Alameda Research is connected to the event
  - `In November 2022 CoinDesk also raised concerns stating that FTX's partner firm Alameda Research held a significant portion of its assets in FTX's native token (FTT).`
  - `On 2 November 2022, CoinDesk published an article stating that Alameda Research, a trading firm affiliated with FTX and owned by FTX chief executive Sam Bankman-Fried, held a significant amount of FTX's exchange token, FTT. The article triggered a spike in withdrawals from FTX`
- ✅ `ftx_bahamas` (present) — The Bahamas appears as FTX's operating jurisdiction
  - `The collapse of cryptocurrency exchange FTX is the subject of scrutiny from government investigators in the Bahamas`
- ✅ `ftx_no_2023_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2023
- ✅ `ftx_no_2026_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2026

## crowdstrike_2024

- Stop: `coverage_reached`; facts 19; episodes 16; duplicate queries 0
- Grounding rejections: 23%; dated facts: 37%; multi-source slots: 50%; claim corroboration: 32%
- Pre-write relevance: 25 accepted, 0 uncertain, 11 rejected; search results rejected: 2; targeted support rounds: 4
- ✅ `crowdstrike_date` (present) — The outage is anchored to July 19, 2024
  - `| Date | 19 July 2024; 2 years ago (2024-07-19) |`
- ✅ `crowdstrike_falcon` (present) — CrowdStrike Falcon is identified as the affected product
  - `On 19 July 2024, CrowdStrike released a faulty configuration update for its Falcon Sensor software on Microsoft Windows systems. The update caused around 8.5 million computers to crash and fail to restart properly.`
  - `An invalid configuration file caused a crash in the CrowdStrike Falcon Sensor code, which runs during the Windows boot process, causing a boot loop.`
- ✅ `crowdstrike_windows` (present) — Windows systems are identified as affected
  - `On July 19, 2024, a defective update from CrowdStrike, an American cybersecurity company, triggered a major IT outage for over 8.5 million devices running the Microsoft system.`
  - `On 19 July 2024, CrowdStrike released a faulty configuration update for its Falcon Sensor software on Microsoft Windows systems. The update caused around 8.5 million computers to crash and fail to restart properly.`
- ✅ `crowdstrike_update` (present) — A faulty content/software update is identified as the trigger
  - `On July 19, 2024, a defective update from CrowdStrike, an American cybersecurity company, triggered a major IT outage for over 8.5 million devices running the Microsoft system.`
  - `In an update on 24th July 2024, CrowdStrike blamed a bug in its quality control procedure.`
- ✅ `crowdstrike_no_2025_event` (absent) — The 2024 outage is not dated to 2025
- ✅ `crowdstrike_no_2026_event` (absent) — The 2024 outage is not dated to 2026

## svb_2023

- Stop: `coverage_reached`; facts 22; episodes 15; duplicate queries 0
- Grounding rejections: 14%; dated facts: 32%; multi-source slots: 50%; claim corroboration: 27%
- Pre-write relevance: 28 accepted, 0 uncertain, 4 rejected; search results rejected: 18; targeted support rounds: 4
- ✅ `svb_closure_date` (present) — SVB's closure is anchored to March 10, 2023
  - `On March 10, 2023, Silicon Valley Bank (SVB) failed after a bank run, marking the third-largest bank failure in United States history and the largest since the 2008 financial crisis.`
- ❌ `svb_fdic` (present) — The FDIC is identified as an intervening authority
- ✅ `svb_bank_run` (present) — Depositor withdrawals or a bank run appear in the mechanism
  - `SVB faced a bank run as depositors rushed to withdraw their funds amid rising inflation and deteriorating financial conditions in the tech sector, where many of its clients operated.`
  - `SVB had heavily invested deposits into long-term treasury bonds, which lost value as the Federal Reserve raised interest rates. Consequently, the bank was unable to meet withdrawal demands, leading to significant losses when it had to sell these bonds at unfavorable prices.`
- ✅ `svb_rates_bonds` (present) — Interest rates and securities losses appear in the explanation
  - `SVB had heavily invested deposits into long-term treasury bonds, which lost value as the Federal Reserve raised interest rates. Consequently, the bank was unable to meet withdrawal demands, leading to significant losses when it had to sell these bonds at unfavorable prices.`
- ✅ `svb_no_2022_closure` (absent) — SVB's closure is not dated to 2022
- ✅ `svb_no_2024_closure` (absent) — SVB's closure is not dated to 2024

## turkiye_quake_2023

- Stop: `all_slots_exhausted`; facts 10; episodes 8; duplicate queries 0
- Grounding rejections: 35%; dated facts: 30%; multi-source slots: 0%; claim corroboration: 0%
- Pre-write relevance: 10 accepted, 0 uncertain, 3 rejected; search results rejected: 76; targeted support rounds: 0
- ✅ `turkiye_quake_date` (present) — The main earthquake is anchored to February 6, 2023
  - `| UTC time | 2023-02-06 01:17:35 |`
- ✅ `turkiye_quake_magnitude` (present) — The mainshock magnitude is identified as Mw 7.8
  - `In the early hours of 6 February 2023, a magnitude 7.8 earthquake struck southern Türkiye. Just 9 hours later, a magnitude 7.5 earthquake hit 90 kilometers (60 miles) to the north.`
  - `In the early hours of 6 February 2023, a magnitude 7.8 earthquake struck southern Türkiye. Just 9 hours later, a magnitude 7.5 earthquake hit 90 kilometers (60 miles) to the north. For weeks afterward, thousands of smaller aftershocks rattled already frail buildings and disrupted relief work.`
- ✅ `turkiye_quake_affected_countries` (present) — Türkiye and Syria are identified as affected countries
  - `The two major 6 February earthquakes (together referred to as the Kahramanmaraş earthquake sequence) occurred on separate faults branching off the 600-kilometer-long (380-mile-long) East Anatolian Fault, which runs through eastern Türkiye and south into Syria, separating the Anatolian and Arabian tectonic plates.`
  - `Two strong earthquakes struck southeast Türkiye and northwest Syria on Feb. 6, killing over 41,000 people.`
- ❌ `turkiye_quake_death_toll` (present) — The death toll is reported at the tens-of-thousands scale
- ✅ `turkiye_quake_rescue_aid` (present) — Rescue or international aid intervention is identified
  - `Project HOPE’s Emergency Response Team is on the ground to support survivors and is working with partners to carry out search and rescue.`
  - `Project HOPE’s partner, SAMU, has K-9 search and rescue teams on the ground in the earthquake zone.`
- ✅ `turkiye_quake_no_wrong_year` (absent) — The February 6 earthquake is not dated to another year
