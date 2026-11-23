# Evidence revalidation under repository changes

## Research question

After a repository changes, which previously observed checks should a coding agent rerun before it commits to a candidate? Measure whether a decision-directed schedule reduces validation cost at a given acceptance rate and whether it recovers a higher-priority candidate whose old failure is obsolete.

## Controlled comparison

Generate executable candidate variants from pinned repository revisions. Require distinct delivered workspaces, independently reproduce the old and new outcomes, and record the actual dependencies used to invalidate observations. Include unchanged inputs, changed dependencies with stable outcomes, failure-to-pass, pass-to-failure, and a still-valid independent blocker. Hold model settings, initial observations, the compacted prefix, test budgets, and the selection rule fixed within comparisons. Record functional acceptance and compatibility regressions separately. [CompAct-Bench](https://arxiv.org/abs/2611.25673) · [SWE-Milestone scoring protocol](https://swe-milestone.com/)

The eager baseline refreshes all stale observed checks when it receives a proposal. The decision-directed schedule excludes a candidate when it has a still-valid failing check. It tries the remaining candidates in the task's declared priority order and stops testing a candidate after a conclusive failure. Both schedules use the same guarded selection and the same public checks. A verified fallback can be returned if budget is exhausted before the higher-priority candidate is resolved. A failure to return a candidate counts against acceptance.

Under deterministic, independent public checks, candidate feasibility is a conjunction. A certificate for selecting the highest-priority feasible candidate includes evidence for every required check of that candidate and one failing check for every higher-priority candidate. Valid cached observations may supply any part of that certificate at zero additional test cost. An offline lower bound can use the reference outcomes to pick the cheapest qualifying failure for each higher-priority candidate. Oracle outcomes belong in analysis only. The online ordering remains a heuristic under unknown outcomes or dependencies. [Bayesian active learning for decision making](https://proceedings.mlr.press/v33/javdani14.html)

## Outcomes and sampling

- Report final acceptance, the highest-priority viable candidate rate, unnecessary retests, and omissions on recovered candidates. Track input and output tokens, phase-specific test executions, CPU time, and end-to-end latency independently. Independent grading costs stay separate.
- Report paired results at fixed test allowances, retaining failed and withheld submissions in the denominator. Aggregate variants within each original defect. Keep repositories intact when splitting development and evaluation data. The pilot uses four defects from two repositories, which supports debugging and descriptive comparisons.
- Audit dependencies by comparing cached fingerprints with freshly measured outcomes. Record a separate result for underdeclared dependencies, inconclusive executions, and changes whose scope the model must discover.
- For a paper-scale run, freeze the sampling rule, quality margin, and budget levels before evaluating held-out repositories. Replicate with another model and 16K context, then test free editing on a milestone sequence. The controlled candidate tasks by themselves establish behavior under predefined alternatives.
