# M5 initial audit — not final acceptance

This directory is the first live three-case run using exact-claim support
metrics. Its purpose was to audit the earlier M4 statement that 94% of slots
were “cross-corroborated.”

That M4 number measured only whether multiple sources contributed any facts to
the same slot. It did not show that the same fact was supported by independent
publishers. With the corrected claim-level metric, this run measured:

| Case | Exact-claim corroboration | High-impact support | Fixed checks |
|---|---:|---:|---:|
| FTX | 2% | 6% | 100% |
| CrowdStrike | 17% | 67% | 100% |
| SVB | 4% | 14% | 83% |

The audit therefore falsified the old interpretation and is intentionally
preserved. It is not the acceptance result for the post-audit M5 code.

After this run, the implementation gained targeted retries for unsupported
exact claims, punctuation-safe entity reuse, stricter primary-actor semantics,
and causal queries that seek both underlying conditions and the immediate
trigger. Those changes require a new SVB run and then a full three-case live
suite before M5 can be marked complete.
