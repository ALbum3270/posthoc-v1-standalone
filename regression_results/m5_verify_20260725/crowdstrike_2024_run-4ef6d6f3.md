# 调查报告：CrowdStrike global IT outage on July 19, 2024

> **覆盖率** 15/15 适用槽位（100%）　**事实** 24 条　**结论级双源支持** 5/24　**来源 episode** 20 个

> 本报告仅消费图谱证据包。每条结论都标注了来源 episode；无证据的槽位如实标记为未查到，不做推断补全。

## WHO

**Primary Actor**
  - The

widespread outage that occurred on Friday 19 July as a result of a CrowdStrike configuration push that put Windows machines into a boot loop may well have been the largest digital systems availability incident that the world has ever seen. `c1b1341d` `87f37023`
    ⚠️ no verified date: this source did not state one explicitly
  来源：[Consequences of Compliance: The CrowdStrike Outage of 19 July 2024 | USENIX](https://www.usenix.org/publications/loginonline/consequences-compliance-crowdstrike-outage-19-july-2024)、[Faulty Configuration Update from CrowdStrike Causes Global Outage | Information Technology Solutions](https://its.ucr.edu/iso-alert/2024/07/30/faulty-configuration-update-crowdstrike-causes-global-outage)

**Affected Parties**
  - An update by cybersecurity firm CrowdStrike led to a major IT outage on Friday, impacting businesses around the world. `33b18224`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Banks and financial companies around the world have reported issues, with German insurance giant Allianz saying it was "experiencing a major outage that is impacting employees' ability to log into their computers. `33b18224`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - German insurance giant Allianz saying it was "experiencing a major outage that is impacting employees' ability to log into their computers. `33b18224`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - NBCUniversal is also being affected by the CrowdStrike outage. `33b18224`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - The National Health Service in England, meanwhile, said it was experiencing disruptions in the majority of doctors' practices. `33b18224`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike suffers major outage affecting businesses around the world](https://www.cnbc.com/2024/07/19/crowdstrike-suffers-major-outage-affecting-businesses-around-the-world.html)

**Related Organizations**
  - Amazon Web Services, eBay, Google Cloud, Instagram, and Plenty of Fish were also affected. `d4dec138`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - In the US, UPS and FedEx were affected. `d4dec138`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Sim racing service iRacing was also affected by the outage in America. `d4dec138`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Various Korean online games, like Black Desert Online, Ragnarok Online, and Ragnarok Origin, shut down. `d4dec138`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)

**Regulators**
  - DHS working with CrowdStrike, others to ‘assess and address’ outage `ecce5c55`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[Microsoft-CrowdStrike issue causes 'largest IT outage in ...](https://www.cnbc.com/2024/07/19/latest-live-updates-on-a-major-it-outage-spreading-worldwide.html)

## WHAT

**Core Event**
  - Cybersecurity firm CrowdStrike (CRWD.O), opens new tab has deployed a fix for an issue that triggered a major tech outage that affected industries ranging from airlines to banking to healthcare worldwide, the company's CEO said on Friday. `dca64207`
    ⚠️ claim has 1 independent source(s); 2 required；no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike deploys fix for issue causing global tech outage](https://www.reuters.com/technology/crowdstrike-says-actively-working-with-customers-impacted-by-outage-2024-07-19)

**Scale**
  - Outages were experienced worldwide, reflecting the wide use of Microsoft Windows and CrowdStrike software by global corporations in numerous business sectors. `86a871c9` `f087e5ac`
    ⚠️ no verified date: this source did not state one explicitly
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)、[Cyber Case Study: CrowdStrike Outage - CoverLink Insurance - Ohio Insurance Agency](https://coverlink.com/cyber-liability-insurance/cyber-case-study-crowdstrike-outage)

**Products**
  - The global outage of specific Microsoft-enabled systems and servers was isolated to a faulty software update released by CrowdStrike `284d964d`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike outage: What you should know](https://www.ibm.com/think/news/recent-crowdstrike-outage-what-you-should-know)

## WHEN

**Event Time**
  - On July 19, 2024, a global IT outage was triggered by a faulty update from CrowdStrike, a leading cybersecurity firm. `8a324932` `3902edcc`
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)、[Understanding the Crowdstrike Outage of July 19th, 2024, and How AI Improves IT Resiliency](https://www.accrete.ai/blog/understanding-the-crowdstrike-outage-of-july-19th-2024-and-how-ai-improves-it-resiliency)

**Discovery Time**
  - July 19, 2024, 05:27 UTC: CrowdStrike identified the issue and reverted the changes, but by then, many systems had already been affected. `eeecb2bc`
    ⚠️ single source; no claim-level corroboration
  来源：[CrowdStrike Outage Timeline, Analysis, & Impact | Bitsight](https://www.bitsight.com/blog/crowdstrike-outage-timeline-and-analysis)

**Intervention Time**
  - Crowdstrike Update - July 23rd
Earlier today, Crowdstrike released an update to the EDR agent to aid in the resolution of the ongoing issues impacting Windows OS-based systems. `fb2b1127`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike Global Outage](https://www.marconet.com/blog/crowdstrike-global-outage)

## WHERE

**Jurisdiction**
  - CrowdStrike Holdings, Inc. is an American cybersecurity technology company based in Austin, Texas. `302e160a`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike](https://en.wikipedia.org/wiki/CrowdStrike)

**Asset Flow** — ⏭️ 不适用：An accidental IT outage normally does not involve movement of money, assets, or activity flows.

**Event Location**
  - The update affected Windows systems, leading to widespread disruptions across multiple sectors, including airlines, healthcare, banking, and public services. `e6d4bac7`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[Understanding the Crowdstrike Outage of July 19th, 2024, and How ...](https://www.accrete.ai/blog/understanding-the-crowdstrike-outage-of-july-19th-2024-and-how-ai-improves-it-resiliency)

## WHY

**Motivation** — ⏭️ 不适用：An accidental IT outage typically has no actor motivation; it is not an intentional action.

**Trigger**
  - On July 19, 2024, a flawed update to CrowdStrike’s Falcon sensor, a cybersecurity tool widely used to protect devices like computer workstations and servers, resulted in a global IT outage. `93e4d720` `6054f28a`
  来源：[Channel File 291 Incident RCA is Available | CrowdStrike](https://www.crowdstrike.com/en-us/blog/channel-file-291-rca-available)、[Monitoring and Support During the CrowdStrike Falcon Outage](https://www.cisecurity.org/insights/case-study/monitoring-and-support-during-the-crowdstrike-falcon-outage)

## HOW

**Mechanism**
  - On July 19, 2024, a global IT outage was triggered by a faulty update from CrowdStrike, a leading cybersecurity firm. `1350a113` `93f192dd`
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)、[Understanding the Crowdstrike Outage of July 19th, 2024, and How ...](https://www.accrete.ai/blog/understanding-the-crowdstrike-outage-of-july-19th-2024-and-how-ai-improves-it-resiliency)

**Sequence**
  - July 19, 2024, 04:09 UTC: CrowdStrike released a sensor configuration update, Channel File 291, to Windows systems as part of their ongoing operations. `932ac27f`
    ⚠️ single source; no claim-level corroboration
  - July 19, 2024, 05:27 UTC: CrowdStrike identified the issue and reverted the changes, but by then, many systems had already been affected. `932ac27f`
    ⚠️ single source; no claim-level corroboration
  - This update triggered a logic error that caused system crashes on impacted machines​. `932ac27f`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike Outage Timeline, Analysis, & Impact](https://www.bitsight.com/blog/crowdstrike-outage-timeline-and-analysis)

---
*由证据包生成。冲突 0 处未消解。无明确日期的事实不带 valid_at（§3.12）。*