# GraphRAG Regression Summary

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Claim corroboration | Critical support | Sources | Rounds | Tokens | Chat cost | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ftx_2022 | 90% | 83% | 100% | 100% | 0% | 0% | 5 | 13 | 30,013 | $0.0185 | 195.1s |
| crowdstrike_2024 | 50% | 83% | 100% | 100% | 0% | 0% | 5 | 12 | 33,276 | $0.0199 | 203.1s |
| svb_2023 | 100% | 83% | 100% | 100% | 0% | 0% | 7 | 24 | 52,285 | $0.0292 | 315.5s |
| turkiye_quake_2023 | 30% | 50% | 100% | 100% | 0% | 0% | 3 | 14 | 18,710 | $0.0112 | 174.4s |

\* Slot relevance is model-judged and is not used for graph writes or stopping.

## ftx_2022

- Stop: `all_slots_exhausted`; facts 17; episodes 9; duplicate queries 0
- Grounding rejections: 48%; dated facts: 18%; multi-source slots: 0%; claim corroboration: 0%
- Pre-write relevance: 19 accepted, 0 uncertain, 6 rejected; search results rejected: 20; targeted support rounds: 0
- ✅ `ftx_bankruptcy_2022` (present) — FTX bankruptcy is anchored to 2022
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
  - `The bankruptcy of FTX, a Bahamas-based cryptocurrency exchange, began in November 2022.`
- ❌ `ftx_sbf` (present) — Sam Bankman-Fried is identified
- ✅ `ftx_alameda` (present) — Alameda Research is connected to the event
  - `On November 11, Alameda Research and FTX declared bankruptcy, and Bankman-Fried stepped down as CEO of FTX.`
- ✅ `ftx_bahamas` (present) — The Bahamas appears as FTX's operating jurisdiction
  - `FTX is incorporated in Antigua and Barbuda and headquartered in the Bahamas.`
  - `FTX is incorporated in Antigua and Barbuda and headquartered in the Bahamas.`
- ✅ `ftx_no_2023_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2023
- ✅ `ftx_no_2026_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2026

## crowdstrike_2024

- Stop: `no_improvement`; facts 10; episodes 5; duplicate queries 0
- Grounding rejections: 59%; dated facts: 30%; multi-source slots: 0%; claim corroboration: 0%
- Pre-write relevance: 11 accepted, 1 uncertain, 10 rejected; search results rejected: 1; targeted support rounds: 0
- ❌ `crowdstrike_date` (present) — The outage is anchored to July 19, 2024
- ✅ `crowdstrike_falcon` (present) — CrowdStrike Falcon is identified as the affected product
  - `On July 19, 2024, CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software, triggering catastrophic`
  - `On 19 July 2024, CrowdStrike released a faulty configuration update for its Falcon Sensor software on Microsoft Windows systems.`
- ✅ `crowdstrike_windows` (present) — Windows systems are identified as affected
  - `“CrowdStrike is actively working with customers impacted by a defect found in a single content update for Windows hosts,” the organization posted on its website early today.`
  - `The outage has been traced back to a cybersecurity company called Crowdstrike which provides anti-cyberattack services to Microsoft, among other companies.`
- ✅ `crowdstrike_update` (present) — A faulty content/software update is identified as the trigger
  - `On July 19, 2024, CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software, triggering catastrophic`
- ✅ `crowdstrike_no_2025_event` (absent) — The 2024 outage is not dated to 2025
- ✅ `crowdstrike_no_2026_event` (absent) — The 2024 outage is not dated to 2026

## svb_2023

- Stop: `coverage_reached`; facts 12; episodes 10; duplicate queries 0
- Grounding rejections: 47%; dated facts: 42%; multi-source slots: 0%; claim corroboration: 0%
- Pre-write relevance: 18 accepted, 0 uncertain, 12 rejected; search results rejected: 31; targeted support rounds: 10
- ✅ `svb_closure_date` (present) — SVB's closure is anchored to March 10, 2023
  - `on Friday March 10, 2023, banking regulators closed California-based Silicon Valley Bank (SVB)`
- ❌ `svb_fdic` (present) — The FDIC is identified as an intervening authority
- ✅ `svb_bank_run` (present) — Depositor withdrawals or a bank run appear in the mechanism
  - `In two days, SVB went from functioning to insolvent when depositors rushed to SVB to withdraw their funds.`
- ✅ `svb_rates_bonds` (present) — Interest rates and securities losses appear in the explanation
  - `The bank’s heavy investment in long-term securities made it highly vulnerable to rising interest rates, leading to substantial unrealized losses.`
- ✅ `svb_no_2022_closure` (absent) — SVB's closure is not dated to 2022
- ✅ `svb_no_2024_closure` (absent) — SVB's closure is not dated to 2024

## turkiye_quake_2023

- Stop: `no_improvement`; facts 5; episodes 3; duplicate queries 0
- Grounding rejections: 64%; dated facts: 20%; multi-source slots: 0%; claim corroboration: 0%
- Pre-write relevance: 6 accepted, 0 uncertain, 2 rejected; search results rejected: 42; targeted support rounds: 0
- ❌ `turkiye_quake_date` (present) — The main earthquake is anchored to February 6, 2023
- ✅ `turkiye_quake_magnitude` (present) — The mainshock magnitude is identified as Mw 7.8
  - `On February 6, 2023, a magnitude 7.8 earthquake occurred in southern Turkey near the northern border of Syria.`
  - `The magnitude-7.8 quake struck on the East Anatolian Fault.`
- ✅ `turkiye_quake_affected_countries` (present) — Türkiye and Syria are identified as affected countries
  - `On February 6, 2023, a magnitude 7.8 earthquake occurred in southern Turkey near the northern border of Syria.`
- ❌ `turkiye_quake_death_toll` (present) — The death toll is reported at the tens-of-thousands scale
- ❌ `turkiye_quake_rescue_aid` (present) — Rescue or international aid intervention is identified
- ✅ `turkiye_quake_no_wrong_year` (absent) — The February 6 earthquake is not dated to another year
