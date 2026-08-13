# Persona Validation

**Status:** Designed scenario, validated against documentary sources. Not a client engagement, and no practitioner interview was conducted.

This document exists so that the user scenario driving this project is auditable — what is grounded in published sources, and what remains inferred.

---

## The scenario

A NYC DOT safety engineer needs to identify where severe and fatal crashes concentrate across the city, in order to develop a targeted safety plan rather than responding reactively after each fatality.

This is a **designed scenario**, chosen to focus the project's scope and give the analysis a concrete decision to serve. It is not based on interviews with NYC DOT staff, and no claim is made that it reflects any specific individual's actual workflow.

## What is documented (grounded in published sources)

**1. Agencies do prioritize locations for safety investment, and the method is published.**
NYC's Vision Zero program publishes borough pedestrian safety action plans that identify priority locations and corridors, with a stated methodology based on crash and injury history. This establishes that location prioritization is a real activity with a real published output — not an invented workflow.

**2. Crash-history-based prioritization is one of two recognized approaches, and the distinction is formal.**
FHWA distinguishes the **hotspot (site-specific) approach**, which selects locations based on their own documented crash history, from the **systemic approach**, which selects locations based on high-risk roadway features correlated with severe crash types, regardless of whether that specific location has a crash record.

**3. Locations without crash history are fundable — through the systemic pathway.**
FHWA developed the systemic approach specifically to address locations that warrant improvement but lack the crash history required to justify funding under hotspot-based selection. Systemic analysis is documented as acceptable justification for Highway Safety Improvement Program (HSIP) grant funding.

**Implication for this project:** this project sits on **both** sides of that line, and the distinction maps cleanly onto its two components.

The Safety Performance Function is a **feature-based systemic model**. Its predictors are `posted_speed`, `number_travel_lanes`, `streetwidth`, `is_highway`, and an imputation flag, plus a log-exposure offset (`src/config.py:181`). There are no crash-derived terms — they were deliberately removed, because feeding crash history into the SPF double-counts the very history Empirical Bayes exists to weigh. A model whose inputs are roadway characteristics, applied network-wide regardless of crash record, is what FHWA describes as the systemic approach.

The Empirical Bayes blend is the **hotspot-side correction**, applied on top for the 13,712 units (6.2%) that have any crash history to correct.

So the accurate description is a **systemic ranking with a hotspot correction**, which is the hybrid the Highway Safety Manual actually prescribes. This matters because the project's headline result depends on it: the +18.4pp lift at N=38,909 comes from ordering the 206,321 zero-history units, which the systemic half can do and a count-based ranking cannot. Restricted to the 13,712 units with history, where both methods have real information, the lift is +2.1pp and does not clear the project's pre-registered 5pp bar. That comparison is published in the README rather than omitted.

**On fundability.** Because the zero-history reach is systemic rather than hotspot, source 3 above is the relevant standard: FHWA documents systemic analysis as acceptable justification for HSIP funding. This partially answers the third unverified question below from documentary sources rather than from an interview. What remains unverified is whether NYC DOT specifically uses that pathway, and what evidence it requires in practice.

**4. SPF + Empirical Bayes is the field's standard method, not an invention of this project.**
The Highway Safety Manual establishes Safety Performance Functions as the standard model form for expected crash frequency, and Empirical Bayes as the standard correction for regression-to-the-mean in site ranking.

### Sources consulted

- [Take Action Before a Crash Occurs: Use a Systemic Approach to Safety — FHWA](https://highways.dot.gov/safety/data-analysis-tools/systemic/take-action-crash-occurs-use-systemic-approach-safety)
- [Applying the Systemic Safety Approach on Local Roads — FHWA](https://highways.dot.gov/safety/data-analysis-tools/systemic/applying-systemic-safety-approach-local-roads)
- [Element 1: The Systemic Safety Planning Process — FHWA](https://highways.dot.gov/safety/data-analysis-tools/systemic/systemic-safety-project-selection-tool/element-1-systemic)
- [Guidance on HSIP MAP-21 Interim Eligibility — FHWA](https://www.fhwa.dot.gov/map21/guidance/guidehsip.cfm)
- NYC Vision Zero borough pedestrian safety action plans (methodology sections)
- AASHTO Highway Safety Manual (SPF and Empirical Bayes methodology)

## What remains unverified

These are stated plainly rather than assumed away:

- **Budget unit.** Whether NYC DOT's safety budget is allocated by individual location, corridor, project type, or another unit is not established here. The application presents rankings at both intersection and corridor level partly for this reason.
- **Approval artifact.** What a DOT engineer must actually produce to move a location from "identified" to "funded" — the specific document, evidence threshold, and internal review path — is not documented in this project.
- **Ranking cadence.** Whether prioritization is revisited annually, quarterly, or continuously is inferred from the published action plans, not confirmed.
- **Whether a re-ranked list would change any real decision.** This project demonstrates that the ranking *changes* under EB adjustment and measures whether the change is predictive. Whether an agency would act on that difference is outside what this analysis can show.
- **Whether NYC DOT specifically uses the systemic pathway.** FHWA documents systemic analysis as acceptable HSIP justification in general. Whether this agency uses it, how often, and what evidence it demands are not established here. This is the single highest-value thing a practitioner conversation would settle, and it is question 3 in `Practitioner_Outreach_Messages.md`.

## Why this is documented rather than omitted

A portfolio project built on an invented user is common; one that states which parts of the user scenario are documented, which are inferred, and which are unverifiable from public sources is less so. The same discipline applies here as in the project's founding finding: name what the obvious version leaves out.

If this project were extended into real practice, the first step would be practitioner interviews on the four unverified items above — not more modeling.
