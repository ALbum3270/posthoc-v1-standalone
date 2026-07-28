# GraphRAG Regression Summary

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Claim corroboration | Critical support | Sources | Rounds | Tokens | Chat cost | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ftx_2022 | 90% | 100% | 100% | 100% | 7% | 17% | 7 | 15 | 35,589 | $0.0205 | 225.0s |
| crowdstrike_2024 | 100% | 83% | 100% | 100% | 13% | 33% | 12 | 24 | 56,137 | $0.0318 | 320.3s |
| svb_2023 | 100% | 50% | 100% | 100% | 12% | 29% | 9 | 22 | 51,868 | $0.0293 | 283.3s |
| turkiye_quake_2023 | 80% | 100% | 100% | 85% | 0% | 0% | 5 | 18 | 27,203 | $0.0166 | 214.3s |

\* Slot relevance is model-judged and is not used for graph writes or stopping.

## ftx_2022

- Stop: `all_slots_exhausted`; facts 14; episodes 10; duplicate queries 0
- Grounding rejections: 51%; dated facts: 29%; multi-source slots: 11%; claim corroboration: 7%
- Pre-write relevance: 15 accepted, 0 uncertain, 6 rejected; search results rejected: 15; targeted support rounds: 0
- ✅ `ftx_bankruptcy_2022` (present) — FTX bankruptcy is anchored to 2022
  - `4 of 4
Full Article
The FTX bankruptcy refers to the collapse of the FTX cryptocurrency exchange, owned by Sam Bankman-Fried, beginning in November 2022.`
  - `4 of 4
Full Article
The FTX bankruptcy refers to the collapse of the FTX cryptocurrency exchange, owned by Sam Bankman-Fried, beginning in November 2022.`
- ✅ `ftx_sbf` (present) — Sam Bankman-Fried is identified
  - `4 of 4
Full Article
The FTX bankruptcy refers to the collapse of the FTX cryptocurrency exchange, owned by Sam Bankman-Fried, beginning in November 2022.`
- ✅ `ftx_alameda` (present) — Alameda Research is connected to the event
  - `In November 2022 CoinDesk also raised concerns stating that FTX's partner firm Alameda Research held a significant portion of its assets in FTX's native token (FTT).`
  - `When did FTX collapse (2022 timeline)
November 2, 2022: CoinDesk investigation exposes Alameda Research's balance sheet dependency on FTT tokens valued at $3.66 billion, revealing dangerous financial entanglement between the trading firm, the exchange, and the price of FTT.`
- ✅ `ftx_bahamas` (present) — The Bahamas appears as FTX's operating jurisdiction
  - `FTX is incorporated in Antigua and Barbuda and headquartered in the Bahamas.`
  - `FTX is incorporated in Antigua and Barbuda and headquartered in the Bahamas.`
- ✅ `ftx_no_2023_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2023
- ✅ `ftx_no_2026_bankruptcy` (absent) — The report does not date FTX's bankruptcy to 2026

## crowdstrike_2024

- Stop: `coverage_reached`; facts 15; episodes 12; duplicate queries 0
- Grounding rejections: 59%; dated facts: 40%; multi-source slots: 20%; claim corroboration: 13%
- Pre-write relevance: 17 accepted, 0 uncertain, 12 rejected; search results rejected: 6; targeted support rounds: 8
- ❌ `crowdstrike_date` (present) — The outage is anchored to July 19, 2024
- ✅ `crowdstrike_falcon` (present) — CrowdStrike Falcon is identified as the affected product
  - `CrowdStrike released technical details that provide:
+ A technical summary of the outage and the impact.
+ Information on how the update to the CrowdStrike Falcon sensor configuration file, Channel File 291, caused the logic error that led to the outage.`
  - `CIS - Center for Internet Security®
CIS Portal LoginCIS Hardened Images®SupportCIS WorkBench Sign In
Alert Level: guarded
HomeInsightsCase StudiesMonitoring and Support During the CrowdStrike Falcon Outage
Monitoring and Support During the CrowdStrike Falcon Outage
On July 19, 2024, a flawed update to CrowdStrike’s Falcon sensor, a cybersecurity tool widely used to protect devices like computer workstations and servers, resulted in a global IT outage.`
- ✅ `crowdstrike_windows` (present) — Windows systems are identified as affected
  - `Faulty CrowdStrike update causes major global IT outage, taking out banks, airlines and businesses globally
CrowdStrike says a fix is on the way and global outage is ‘not a cyberattack’
Businesses across the world are reporting IT outages, including Windows “blue screen of death” errors on their computers, in what has already become one of the most widespread IT disruptions in recent years.`
  - `The outage — linked to a software update from popular cybersecurity firm CrowdStrike — has affected computers running Microsoft Windows at organizations across various sectors, including airlines, banks, retailers, brokerage houses, media companies and railway networks.`
- ✅ `crowdstrike_update` (present) — A faulty content/software update is identified as the trigger
  - `Crowdstrike Outage in Numbers
The outage was caused by a defect in a Falcon content update for Windows hosts. Specifically, the update was related to Channel File 291, which controls how Falcon evaluates named pipe execution on Windows systems.`
- ✅ `crowdstrike_no_2025_event` (absent) — The 2024 outage is not dated to 2025
- ✅ `crowdstrike_no_2026_event` (absent) — The 2024 outage is not dated to 2026

## svb_2023

- Stop: `coverage_reached`; facts 17; episodes 12; duplicate queries 0
- Grounding rejections: 52%; dated facts: 47%; multi-source slots: 20%; claim corroboration: 12%
- Pre-write relevance: 19 accepted, 0 uncertain, 9 rejected; search results rejected: 5; targeted support rounds: 11
- ❌ `svb_closure_date` (present) — SVB's closure is anchored to March 10, 2023
- ❌ `svb_fdic` (present) — The FDIC is identified as an intervening authority
- ✅ `svb_bank_run` (present) — Depositor withdrawals or a bank run appear in the mechanism
  - `| This article is part of a series about the |
| 2023 United States banking crisis |
| Press camera outside a Signature Bank branch in New York on March 13, 2023 |
| Background Economic Growth, Regulatory Relief and Consumer Protection Act 2021–2023 inflation surge 2020–2022 cryptocurrency bubble + Bankruptcy of FTX |
| Events Winding down of Silvergate Bank Collapse of Silicon Valley Bank Collapse of Signature Bank Collapse of First Republic Bank |
| Related groups Federal Reserve Board of Governors Federal Deposit Insurance Corporation United States Department of the Treasury |
| Effects Bank Term Funding Program Acquisition of Credit Suisse by UBS |
| v t e |
On March 10, 2023, Silicon Valley Bank (SVB) failed after a bank run, marking the third-largest bank failure in United States history and the largest since the 2008 financial crisis.`
- ✅ `svb_rates_bonds` (present) — Interest rates and securities losses appear in the explanation
  - `However, the SVB DST did not shift its attention to the rising interest rates and SVB’s unrealized losses on investment securities and did not sufficiently assess the extent of the potential effect to the bank’s capital and liquidity.`
- ❌ `svb_no_2022_closure` (absent) — SVB's closure is not dated to 2022
  - `| This article is part of a series about the |
| 2023 United States banking crisis |
| Press camera outside a Signature Bank branch in New York on March 13, 2023 |
| Background Economic Growth, Regulatory Relief and Consumer Protection Act 2021–2023 inflation surge 2020–2022 cryptocurrency bubble + Bankruptcy of FTX |
| Events Winding down of Silvergate Bank Collapse of Silicon Valley Bank Collapse of Signature Bank Collapse of First Republic Bank |
| Related groups Federal Reserve Board of Governors Federal Deposit Insurance Corporation United States Department of the Treasury |
| Effects Bank Term Funding Program Acquisition of Credit Suisse by UBS |
| v t e |
On March 10, 2023, Silicon Valley Bank (SVB) failed after a bank run, marking the third-largest bank failure in United States history and the largest since the 2008 financial crisis.`
  - `| This article is part of a series about the |
| 2023 United States banking crisis |
| Press camera outside a Signature Bank branch in New York on March 13, 2023 |
| Background Economic Growth, Regulatory Relief and Consumer Protection Act 2021–2023 inflation surge 2020–2022 cryptocurrency bubble + Bankruptcy of FTX |
| Events Winding down of Silvergate Bank Collapse of Silicon Valley Bank Collapse of Signature Bank Collapse of First Republic Bank |
| Related groups Federal Reserve Board of Governors Federal Deposit Insurance Corporation United States Department of the Treasury |
| Effects Bank Term Funding Program Acquisition of Credit Suisse by UBS |
| v t e |
On March 10, 2023, Silicon Valley Bank (SVB) failed after a bank run, marking the third-largest bank failure in United States history and the largest since the 2008 financial crisis.`
- ✅ `svb_no_2024_closure` (absent) — SVB's closure is not dated to 2024

## turkiye_quake_2023

- Stop: `all_slots_exhausted`; facts 13; episodes 8; duplicate queries 0
- Grounding rejections: 31%; dated facts: 23%; multi-source slots: 0%; claim corroboration: 0%
- Pre-write relevance: 13 accepted, 0 uncertain, 9 rejected; search results rejected: 58; targeted support rounds: 0
- ✅ `turkiye_quake_date` (present) — The main earthquake is anchored to February 6, 2023
  - `An Eye on the Mediterranean
Cover of the September 2023 issue of Eos
The 2023 Türkiye-Syria Earthquakes Shifted Stress in the Crust
A Common Language for Reporting Earthquake Intensities
How Hail Hazards Are Changing Around the Mediterranean
Protecting the Mountain Water Towers of Spain’s Sierra Nevada
Beyond the Wine-Dark Sea
In the early hours of 6 February 2023, a magnitude 7.8 earthquake struck southern Türkiye.`
- ✅ `turkiye_quake_magnitude` (present) — The mainshock magnitude is identified as Mw 7.8
  - `An Eye on the Mediterranean
Cover of the September 2023 issue of Eos
The 2023 Türkiye-Syria Earthquakes Shifted Stress in the Crust
A Common Language for Reporting Earthquake Intensities
How Hail Hazards Are Changing Around the Mediterranean
Protecting the Mountain Water Towers of Spain’s Sierra Nevada
Beyond the Wine-Dark Sea
In the early hours of 6 February 2023, a magnitude 7.8 earthquake struck southern Türkiye. Just 9 hours later, a magnitude 7.5 earthquake hit 90 kilometers (60 miles) to the north.`
  - `An Eye on the Mediterranean
Cover of the September 2023 issue of Eos
The 2023 Türkiye-Syria Earthquakes Shifted Stress in the Crust
A Common Language for Reporting Earthquake Intensities
How Hail Hazards Are Changing Around the Mediterranean
Protecting the Mountain Water Towers of Spain’s Sierra Nevada
Beyond the Wine-Dark Sea
In the early hours of 6 February 2023, a magnitude 7.8 earthquake struck southern Türkiye.`
- ✅ `turkiye_quake_affected_countries` (present) — Türkiye and Syria are identified as affected countries
  - `The two major 6 February earthquakes (together referred to as the Kahramanmaraş earthquake sequence) occurred on separate faults branching off the 600-kilometer-long (380-mile-long) East Anatolian Fault, which runs through eastern Türkiye and south into Syria, separating the Anatolian and Arabian tectonic plates.`
  - `The earthquakes and underlying vulnerabilities resulted in the deaths of at least 56,000 people in Turkey and Syria.`
- ✅ `turkiye_quake_death_toll` (present) — The death toll is reported at the tens-of-thousands scale
  - `The earthquakes and underlying vulnerabilities resulted in the deaths of at least 56,000 people in Turkey and Syria.`
- ✅ `turkiye_quake_rescue_aid` (present) — Rescue or international aid intervention is identified
  - `Partnership with Turkish government
[edit]
Following the 2016 Turkish coup d'état attempt, the Turkish Red Crescent backed the Turkish government, sending a letter to hundreds of international aid organizations and NGOs, including to organizations of the United Nations and Red Crescents in 191 total countries.`
  - `Turkish Red Crescent
| Türk Kızılay |
| Logo of the Turkish Red Crescent |
| Formation | 1868; 158 years ago (1868) |
| Founded at | Ottoman Empire (1868) re-established in Ankara, Turkey (1935) |
| Type | Humanitarian Aid |
| Legal status | Active; auxiliary role to the Turkish government and public authorities in emergency and humanitarian operations |
| Purpose | Disaster relief; emergency medical aid; refugee support; blood donation; healthcare; social services; international humanitarian aid |
| Headquarters | Ankara, Turkey |
| Members | International Federation of Red Cross and Red Crescent Societies (IFRC) |
| Official language | Turkish |
| President | Fatma Meriç Yılmaz |
| Staff | 6,423 (headquarters & branch employees, 2023) |
| Volunteers | 327,114 (2023) |
| Website | www.kizilay.org.tr |
The Turkish Red Crescent (Turkish: Türk Kızılay) is the Turkish affiliate of the`
- ✅ `turkiye_quake_no_wrong_year` (absent) — The February 6 earthquake is not dated to another year
