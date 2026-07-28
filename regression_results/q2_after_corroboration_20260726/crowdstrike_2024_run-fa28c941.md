# 调查报告：CrowdStrike global IT outage on July 19, 2024

> **覆盖率** 10/10 必填槽位（100%）　**选填槽** 0/5（不计入覆盖率）　**事实** 15 条　**结论级双源支持** 2/15　**来源 episode** 12 个

> 本报告仅消费图谱证据包。每条结论都标注了来源 episode；无证据的槽位如实标记为未查到，不做推断补全。

## WHO

**Primary Actor**
  - CrowdStrike released technical details that provide:
+ A technical summary of the outage and the impact.
+ Information on how the update to the CrowdStrike Falcon sensor configuration file, Channel File 291, caused the logic error that led to the outage. `0a734ada` `e3ee3caa`
    ⚠️ no verified date: this source did not state one explicitly
  来源：[(PDF) CrowdStrike Causes Global Microsoft Outage: A Case Study](https://www.researchgate.net/publication/393170974_CrowdStrike_Causes_Global_Microsoft_Outage_A_Case_Study)、[Widespread IT Outage Due to CrowdStrike Update](https://www.cisa.gov/news-events/alerts/2024/07/19/widespread-it-outage-due-crowdstrike-update)

**Affected Parties**
  - Faulty CrowdStrike update causes major global IT outage, taking out banks, airlines and businesses globally
CrowdStrike says a fix is on the way and global outage is ‘not a cyberattack’
Businesses across the world are reporting IT outages, including Windows “blue screen of death” errors on their computers, in what has already become one of the most widespread IT disruptions in recent years. `fc5a84f4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Many companies’ employees have reported being unable to start their computers due to the issue. `fc5a84f4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - The outage — linked to a software update from popular cybersecurity firm CrowdStrike — has affected computers running Microsoft Windows at organizations across various sectors, including airlines, banks, retailers, brokerage houses, media companies and railway networks. `fc5a84f4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - The travel sector seems to be one of the hardest hit, based on online chatter. `fc5a84f4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - U.K. news broadcaster Sky News faced trouble broadcasting live this morning due to the outage, the firm’s executive chairman David Rhodes tweeted. `fc5a84f4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[Faulty CrowdStrike update causes major global IT outage, taking out banks, airlines and businesses globally | TechCrunch](https://techcrunch.com/2024/07/19/faulty-crowdstrike-update-causes-major-global-it-outage-taking-out-banks-airlines-and-businesses-globally)

**Related Organizations**
  - Understanding the Crowdstrike Outage of July 19th, 2024, and How AI Improves IT Resiliency
Overview of the Incident
On July 19, 2024, a global IT outage was triggered by a faulty update from CrowdStrike, a leading cybersecurity firm. `ad7786cd`
    ⚠️ single source; no claim-level corroboration
  来源：[Understanding the Crowdstrike Outage of July 19th, 2024 ...](https://www.accrete.ai/blog/understanding-the-crowdstrike-outage-of-july-19th-2024-and-how-ai-improves-it-resiliency)

**Regulators** — ⚠️ 未查到相关证据

## WHAT

**Core Event**
  - CIS - Center for Internet Security®
CIS Portal LoginCIS Hardened Images®SupportCIS WorkBench Sign In
Alert Level: guarded
HomeInsightsCase StudiesMonitoring and Support During the CrowdStrike Falcon Outage
Monitoring and Support During the CrowdStrike Falcon Outage
On July 19, 2024, a flawed update to CrowdStrike’s Falcon sensor, a cybersecurity tool widely used to protect devices like computer workstations and servers, resulted in a global IT outage. `087b95c3`
    ⚠️ claim has 1 independent source(s); 2 required；single source; no claim-level corroboration
  来源：[Monitoring and Support During the CrowdStrike Falcon Outage](https://www.cisecurity.org/insights/case-study/monitoring-and-support-during-the-crowdstrike-falcon-outage)

**Scale**
  - Microsoft estimates that this event affected 8.5 million systems, which is less than 1% of total Windows machines. `f25c126a`
    ⚠️ claim has 1 independent source(s); 2 required；no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[The July 19th Global IT Outages](https://www.congress.gov/crs_external_products/IN/PDF/IN12392/IN12392.2.pdf)

**Products** — ⚠️ 未查到相关证据

## WHEN

**Event Time**
  - The CrowdStrike outage in July 2024 exposed how much the world depends on centralized security solutions. `d8855cdf`
    ⚠️ claim has 1 independent source(s); 2 required；single source; no claim-level corroboration
  来源：[What We Can Learn from the 2024 CrowdStrike Outage](https://cloudsecurityalliance.org/blog/2025/07/03/what-we-can-learn-from-the-2024-crowdstrike-outage)

**Discovery Time** — ⚠️ 未查到相关证据

**Intervention Time** — ⚠️ 未查到相关证据

## WHERE

**Jurisdiction** — ⚠️ 未查到相关证据

**Asset Flow** — ⏭️ 不适用：An accidental IT outage normally has no asset or money flow involved.

**Event Location**
  - — Kif Leswing
Fri, Jul 19 2024 5:04 PM EDT
Amazon warehouses and internal software disrupted by outage
Peter Endig | AFP | Getty Images
Some Amazon warehouses in the U.S. were grappling with disruptions set off by the global IT outage. `4cb3f25a`
    ⚠️ single source; no claim-level corroboration
  来源：[Global IT outage live updates: Microsoft-CrowdStrike blackout](https://www.cnbc.com/amp/2024/07/19/latest-live-updates-on-a-major-it-outage-spreading-worldwide.html)

## WHY

**Motivation** — ⏭️ 不适用：An accidental IT outage typically has no actor motivation behind it.

**Trigger**
  - The root cause analysis highlighted several factors contributing to the Falcon EDR sensor crash. These included a mismatch between inputs validated by a Content Validator and those provided to a Content Interpreter, an out-of-bounds read issue in the Content Interpreter, and the absence of a specific test. `05338f2e` `e0889428`
    ⚠️ no verified date: this source did not state one explicitly
  来源：[CrowdStrike Releases Root Cause Analysis (RCA) Report of Global IT Outage](https://www.linkedin.com/pulse/crowdstrike-releases-root-cause-analysis-rca-report-mleac)、[CrowdStrike Releases Root Cause Analysis of Falcon Sensor BSOD Crash - SecurityWeek](https://www.securityweek.com/crowdstrike-releases-root-cause-analysis-of-falcon-sensor-bsod-crash/amp)

## HOW

**Mechanism**
  - Crowdstrike Outage in Numbers
The outage was caused by a defect in a Falcon content update for Windows hosts. Specifically, the update was related to Channel File 291, which controls how Falcon evaluates named pipe execution on Windows systems. `21757b0a`
    ⚠️ claim has 1 independent source(s); 2 required；no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[The CrowdStrike Incident: A Global IT Meltdown](https://www.blackfog.com/the-crowdstrike-incident-a-global-it-meltdown)

**Sequence**
  - According to CrowdStrike and other sources, the timeline of events was as follows:
July 19, 2024, 04:09 UTC: CrowdStrike released a sensor configuration update, Channel File 291, to Windows systems as part of their ongoing operations. `33314517`
    ⚠️ single source; no claim-level corroboration
  - July 19, 2024, 05:27 UTC: CrowdStrike identified the issue and reverted the changes, but by then, many systems had already been affected. `33314517`
    ⚠️ single source; no claim-level corroboration
  来源：[CrowdStrike Outage Timeline, Analysis, & Impact | Bitsight](https://www.bitsight.com/blog/crowdstrike-outage-timeline-and-analysis)

---
*由证据包生成。冲突 0 处未消解。无明确日期的事实不带 valid_at（§3.12）。*