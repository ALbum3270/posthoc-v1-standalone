# 调查报告：CrowdStrike global IT outage on July 19, 2024

> **覆盖率** 10/15 适用槽位（100%）　**事实** 19 条　**结论级双源支持** 6/19　**来源 episode** 16 个

> 本报告仅消费图谱证据包。每条结论都标注了来源 episode；无证据的槽位如实标记为未查到，不做推断补全。

## WHO

**Primary Actor**
  - CrowdStrike's major IT outage, which has affected businesses globally, is leading the stock to its worst weekly performance since November 2022. `db0479b3` `0705ac2f`
  来源：[Global IT outage live updates: Microsoft-CrowdStrike blackout](https://www.cnbc.com/amp/2024/07/19/latest-live-updates-on-a-major-it-outage-spreading-worldwide.html)、[Understanding the Crowdstrike Outage of July 19th, 2024, and How AI Improves IT Resiliency](https://www.accrete.ai/blog/understanding-the-crowdstrike-outage-of-july-19th-2024-and-how-ai-improves-it-resiliency)

**Affected Parties**
  - A non-malicious global technology outage that began in the early morning of July 19 is continuing to affect many industries `b641d55b`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Impact is also being felt indirectly as a result of local emergency call centers being down `b641d55b`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - These disruptions are resulting in some clinical procedure delays, diversions or cancellations `b641d55b`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - is having varying effects on hospitals and health systems across the country `b641d55b`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - we are hearing from hospitals and health systems that the impact varies widely. Some have experienced little to no impact while others are dealing directly with some disruptions to medical technology, communications and third-party service providers `b641d55b`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike technology outage causing global disruption across industries, including impacts on hospitals and health systems | AHA News](https://www.aha.org/news/headline/2024-07-19-crowdstrike-technology-outage-causing-global-disruption-across-industries-including-impacts-hospitals)

**Related Organizations**
  - An update by cybersecurity firm CrowdStrike led to a major IT outage on Friday, impacting businesses around the world. `b3ed2153`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike suffers major outage affecting businesses around the world](https://www.cnbc.com/2024/07/19/crowdstrike-suffers-major-outage-affecting-businesses-around-the-world.html)

**Regulators** — ⚠️ 未查到相关证据

## WHAT

**Core Event**
  - On July 19, 2024, a defective update from CrowdStrike, an American cybersecurity company, triggered a major IT outage for over 8.5 million devices running the Microsoft system. `785c8cd8`
    ⚠️ claim has 1 independent source(s); 2 required；single source; no claim-level corroboration
  来源：[July 19, 2024 Incident : When an Update Has Global Impacts](https://www.premiercontinuum.com/resources/microsoft-outbreak-july-2024)

**Scale**
  - On 19 July 2024, CrowdStrike released a faulty configuration update for its Falcon Sensor software on Microsoft Windows systems. The update caused around 8.5 million computers to crash and fail to restart properly. `ba99b34e` `cfe2c699`
  来源：[CrowdStrike](https://en.wikipedia.org/wiki/CrowdStrike)、[Crowdstrike Reviews, News, and Deals](https://www.pcmag.com/brands/crowdstrike)

**Products** — ⚠️ 未查到相关证据

## WHEN

**Event Time**
  - | Date | 19 July 2024; 2 years ago (2024-07-19) | `69099b12` `f421af43`
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)、[Understanding the Crowdstrike Outage of July 19th, 2024, and How ...](https://www.accrete.ai/blog/understanding-the-crowdstrike-outage-of-july-19th-2024-and-how-ai-improves-it-resiliency)

**Discovery Time** — ⚠️ 未查到相关证据

**Intervention Time** — ⚠️ 未查到相关证据

## WHERE

**Jurisdiction** — ⚠️ 未查到相关证据

**Asset Flow** — ⏭️ 不适用：An accidental IT outage normally has no asset or money flow involved.

**Event Location**
  - Multiple blue screens of death caused by a faulty software update on baggage carousels at LaGuardia Airport, New York City `023a578e`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Outages were experienced worldwide, reflecting the wide use of Microsoft Windows and CrowdStrike software by global corporations in numerous business sectors. `023a578e`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  - Widespread outages were immediately reported across multiple countries, with major global disturbances experienced by the general public. `023a578e`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)

## WHY

**Motivation** — ⏭️ 不适用：An accidental IT outage typically has no actor motivation.

**Trigger**
  - CrowdStrike stressed that the IT outage was not caused by a cyber-attack or any other criminal activity – but, as a failure to apply proper safeguards to a critical patch, it was a major security incident. `a3b0d1f0` `7af24523`
    ⚠️ no verified date: this source did not state one explicitly
  - In an update on 24th July 2024, CrowdStrike blamed a bug in its quality control procedure. `a3b0d1f0` `7f235d2f`
  来源：[2024 CrowdStrike-related IT outages](https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages)、[CrowdStrike](https://en.wikipedia.org/wiki/CrowdStrike)、[Reflections on the CrowdStrike Global IT Outage](https://www.quorumcyber.com/insights/reflections-on-the-crowdstrike-global-it-outage)

## HOW

**Mechanism**
  - An invalid configuration file caused a crash in the CrowdStrike Falcon Sensor code, which runs during the Windows boot process, causing a boot loop. `774c166e` `4844b135`
    ⚠️ no verified date: this source did not state one explicitly
  来源：[Consequences of Compliance: The CrowdStrike Outage of 19 July 2024 | USENIX](https://www.usenix.org/publications/loginonline/consequences-compliance-crowdstrike-outage-19-july-2024)、[CrowdStrike Outage Explained](https://www.youtube.com/watch?v=9m_YXDrRkic)

**Sequence**
  - July 19, 2024, 04:09 UTC: CrowdStrike released a sensor configuration update, Channel File 291, to Windows systems as part of their ongoing operations. `18d4effc`
    ⚠️ single source; no claim-level corroboration
  - July 19, 2024, 05:27 UTC: CrowdStrike identified the issue and reverted the changes, but by then, many systems had already been affected. `18d4effc`
    ⚠️ single source; no claim-level corroboration
  - This update triggered a logic error that caused system crashes on impacted machines​. `18d4effc`
    ⚠️ no verified date: this source did not state one explicitly；single source; no claim-level corroboration
  来源：[CrowdStrike Outage Timeline, Analysis, & Impact](https://www.bitsight.com/blog/crowdstrike-outage-timeline-and-analysis)

---
*由证据包生成。冲突 0 处未消解。无明确日期的事实不带 valid_at（§3.12）。*