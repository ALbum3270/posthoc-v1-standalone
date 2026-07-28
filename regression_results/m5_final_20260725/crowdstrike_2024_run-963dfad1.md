# 调查报告：CrowdStrike global IT outage on July 19, 2024

> **覆盖率** 15/15 适用槽位（100%）　**事实** 20 条　**结论级双源支持** 4/20　**来源 episode** 19 个

> 本报告仅消费图谱证据包。每条结论都标注了来源 episode；无证据的槽位如实标记为未查到，不做推断补全。

## WHO

**Primary Actor**
  - Information on how the update to the CrowdStrike Falcon sensor configuration file, Channel File 291, caused the logic error that led to the outage. `b523ec14` `80e060d0`
    ⚠️ no verified date: this source did not state one explicitly
  来源：[(PDF) CrowdStrike Causes Global Microsoft Outage: A Case Study](https://www.researchgate.net/publication/393170974_CrowdStrike_Causes_Global_Microsoft_Outage_A_Case_Study)、[Widespread IT Outage Due to CrowdStrike Update](https://www.cisa.gov/news-events/alerts/2024/07/19/widespread-it-outage-due-crowdstrike-update)

**Affected Parties**
  - The update triggered a memory error that led to system crashes and widespread Blue Screen of Death (BSOD) issues at private and public sector organizations globally. `bf73f3d6`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[Monitoring and Support During the CrowdStrike Falcon Outage](https://www.cisecurity.org/insights/case-study/monitoring-and-support-during-the-crowdstrike-falcon-outage)

**Related Organizations**
  - Amazon Web Services, eBay, Google Cloud, Instagram, and Plenty of Fish were also affected. `705612a4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - In the US, UPS and FedEx were affected. `705612a4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Sim racing service iRacing was also affected by the outage in America. `705612a4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Various Korean online games, like Black Desert Online, Ragnarok Online, and Ragnarok Origin, shut down. `705612a4`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)

**Regulators**
  - DHS working with CrowdStrike, others to ‘assess and address’ outage `04ceb3cc`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[Global IT outage live updates: Microsoft-CrowdStrike blackout](https://www.cnbc.com/2024/07/19/latest-live-updates-on-a-major-it-outage-spreading-worldwide.html)

## WHAT

**Core Event**
  - On July 19, 2024, a defective update from CrowdStrike, an American cybersecurity company, triggered a major IT outage for over 8.5 million devices running the Microsoft system. `728254c0`
    ⚠️ claim has 1 independent source(s); 2 required；single source; no claim-level corroboration
  来源：[July 19, 2024 Incident : When an Update Has Global Impacts](https://www.premiercontinuum.com/resources/microsoft-outbreak-july-2024)

**Scale**
  - In July 2024, millions of Windows users were locked out of their systems due to a flaw in a CrowdStrike update. `a05ac30f` `20f3c7e0`
  来源：[Incident 6 - CrowdStrike (2024) : The Update That Took Millions of Windows Hosts Down](https://www.linkedin.com/pulse/incident-6-crowdstrike-2024-update-took-millions-emad-m-abdelhamid-0wgpf)、[The Lasting Impact of the CrowdStrike Update Outage | Tufin](https://www.tufin.com/blog/lasting-impact-of-crowdstrike-update-outage)

**Products**
  - a faulty software update released by CrowdStrike `5d44a80d`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike outage: What you should know](https://www.ibm.com/think/news/recent-crowdstrike-outage-what-you-should-know)

## WHEN

**Event Time**
  - | Date | 19 July 2024; 2 years ago (2024-07-19) | `c2000705` `4b06a0e7`
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)、[July 19, 2024 Incident : When an Update Has Global Impacts](https://www.premiercontinuum.com/resources/microsoft-outbreak-july-2024)

**Discovery Time**
  - July 19, 2024, 05:27 UTC: CrowdStrike identified the issue and reverted the changes, but by then, many systems had already been affected. `c5b0bdf4`
    ⚠️ single source; no claim-level corroboration
  来源：[CrowdStrike Outage Timeline, Analysis, & Impact](https://www.bitsight.com/blog/crowdstrike-outage-timeline-and-analysis)

**Intervention Time**
  - Update 9:45 a.m., EDT, July 21, 2024:
Microsoft released a recovery tool that uses a USB drive to boot and repair affected systems. `fb543019`
    ⚠️ single source; no claim-level corroboration
  来源：[Widespread IT Outage Due to CrowdStrike Update](https://www.cisa.gov/news-events/alerts/2024/07/19/widespread-it-outage-due-crowdstrike-update)

## WHERE

**Jurisdiction**
  - CrowdStrike Holdings, Inc. is an American cybersecurity technology company based in Austin, Texas. `9cb53cfe`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike](https://en.wikipedia.org/wiki/CrowdStrike)

**Asset Flow** — ⏭️ 不适用：An accidental IT outage typically does not involve movement of money, assets, or activity flows.

**Event Location**
  - causing a major outage of businesses and services worldwide. `fa4eb649`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike outage: What you should know](https://www.ibm.com/think/news/recent-crowdstrike-outage-what-you-should-know)

## WHY

**Motivation** — ⏭️ 不适用：An accidental IT outage normally has no actor motivation or intentional behavior behind it.

**Trigger**
  - On July 19, 2024, CrowdStrike, a leading cybersecurity firm, deployed a faulty update for its Falcon Sensor software `5885b69e`
    ⚠️ claim has 1 independent source(s); 2 required；single source; no claim-level corroboration
  来源：[CrowdStrike Causes Global Microsoft Outage: A Case Study](https://www.researchgate.net/publication/393170974_CrowdStrike_Causes_Global_Microsoft_Outage_A_Case_Study)

## HOW

**Mechanism**
  - The update caused system crashes, commonly referred to as the "Blue Screen of Death" (BSOD), due to a logic error in the update’s configuration file for the Falcon sensor version 7.11 and above `17846bbe` `0f27625a`
    ⚠️ no verified date: this source did not state one explicitly
  来源：[CrowdStrike blames outage on content configuration update | Computer Weekly](https://www.computerweekly.com/news/366598755/CrowdStrike-blames-outage-on-content-configuration-update)、[Understanding the Crowdstrike Outage of July 19th, 2024, and How AI Improves IT Resiliency](https://www.accrete.ai/blog/understanding-the-crowdstrike-outage-of-july-19th-2024-and-how-ai-improves-it-resiliency)

**Sequence**
  - July 19, 2024, 04:09 UTC: CrowdStrike released a sensor configuration update, Channel File 291, to Windows systems as part of their ongoing operations. `21ffcd49`
    ⚠️ single source; no claim-level corroboration
  - July 19, 2024, 05:27 UTC: CrowdStrike identified the issue and reverted the changes, but by then, many systems had already been affected. `21ffcd49`
    ⚠️ single source; no claim-level corroboration
  - This update triggered a logic error that caused system crashes on impacted machines​. `21ffcd49`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike Outage Timeline, Analysis, & Impact | Bitsight](https://www.bitsight.com/blog/crowdstrike-outage-timeline-and-analysis)

---
*由证据包生成。冲突 0 处未消解。无明确日期的事实不带 valid_at（§3.12）。*