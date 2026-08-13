# Practitioner Outreach — NYC Crash Prioritization

**Goal:** one 20-minute conversation with someone who has actually produced or used a priority-location list for NYC street safety funding.

**The three questions that need answering:**

1. What unit is the budget in? (Locations? Corridors? Project types? Linear feet?)
2. What must you produce to get a location funded? (What's the actual artifact and evidence standard?)
3. If you were handed twenty locations with little or no crash history but a high predicted risk, would that be usable or unfundable?

Question 3 is the decisive one. It determines whether an Empirical Bayes–adjusted ranking — which by design surfaces locations whose history understates their risk — is aimed at a real gap or at something the funding process structurally cannot act on.

---

## Version A — LinkedIn / cold DM (short)

Subject line (if email): A data finding about NYC crash records — and one question I can't answer from the data

> Hi [Name] — I'm working with the NYC Open Data motor vehicle collisions dataset and found something I think is worth flagging: the borough field is NULL on about 30% of crash rows, and those rows account for roughly 40% of all traffic deaths. They're not random — they're concentrated on the Belt Parkway, LIE, BQE, Grand Central, FDR, and the Cross Bronx. Limited-access highways that fall outside the precinct street-grid geocoding. So the standard "crashes by borough" chart most dashboards open with is biased toward surface streets and drops the roads where crashes are most likely to kill someone.
>
> I'm now looking at whether a Safety Performance Function with an Empirical Bayes adjustment would rank priority locations differently than a historical-count ranking does. But I've hit a question the data can't answer, and you would know it cold:
>
> **If someone handed you twenty locations with little or no crash history but a high predicted risk, would that list be usable — or unfundable?**
>
> I'd value 20 minutes if you're open to it. Happy to share the analysis either way.
>
> — Jeff

## Version B — Email to a consultancy analyst or agency staffer (slightly fuller)

Subject: Question about priority-location funding criteria (NYC crash data analysis)

> Hi [Name],
>
> I'm a data analyst working through NYC's Motor Vehicle Collisions dataset (Socrata h9gi-nx95, ~2.27M rows, 2012–2026). One finding so far: the borough field is missing on 30.5% of rows, and those unlabeled rows carry 39.8% of all traffic deaths — a fatality rate about 1.5x the labeled rows. The missing rows cluster on limited-access highways (Belt Parkway, LIE, BQE, Grand Central, FDR, Cross Bronx, Major Deegan), which sit outside the precinct geocoding that assigns a borough. The practical implication is that the standard borough breakdown systematically under-represents the deadliest roads.
>
> I'm extending this into a prioritization question: does an SPF + Empirical Bayes ranking surface different locations than a historical-count ranking, and does that difference hold up against what actually happened in the following 12 months?
>
> Before I build further, I want to make sure I'm aiming at a real decision rather than an assumed one. Three questions, if you have 20 minutes:
>
> 1. What unit does the safety budget actually work in — individual locations, corridors, project types, something else?
> 2. What do you have to produce to get a location funded? What's the evidence standard?
> 3. If you received twenty locations with minimal crash history but high predicted risk, would that be actionable, or would it fail the justification requirement?
>
> That third one especially — it determines whether the method I'm using is solving a real gap or an imaginary one, and I'd rather find that out now than after building it.
>
> I'm happy to share the full analysis and code regardless of whether you have time to talk.
>
> Thanks for considering it,
> Jeff
> [GitHub link] · [email]

## Version C — In person (ITE event, Vision Zero workshop, meetup)

Opener, spoken:

> "I've been working through the city's crash dataset and found that about 40% of traffic deaths sit in rows where the borough field is blank — mostly Belt Parkway, the LIE, the BQE. So the standard borough chart misses the deadliest roads entirely. I'm trying to figure out whether a risk model would prioritize differently than a crash-count ranking. Can I ask you something I can't get from the data — if you got handed twenty locations that had almost no crash history but scored high on predicted risk, could you actually fund those, or would they get bounced for lack of justification?"

Then stop talking and listen. The answer to that question is the deliverable.

---

## Practical notes

- **Lead with the finding, not the ask.** You're bringing something to the conversation, and it establishes immediately that you've done real work.
- **Don't ask for career help.** This is a methodology question between people who work with data. That framing gets far better response rates and better answers.
- **Ask question 3 explicitly, even if the conversation goes well elsewhere.** It's easy to have a pleasant 20 minutes and come away without the one answer that changes the design.
- **Record the answer verbatim** in the project repo (a `docs/persona-validation.md`), including the date and the person's role. That converts "inferred persona" into "validated persona" in the project narrative — and a hiring manager who sees you went and asked will weight the whole project differently.
- **If the answer is "unfundable,"** that's not a failed project — it's a finding. The pivot would be toward locations with history that the incumbent ranking still misranks, which is still a real and defensible audit.

## Where to send it

| Channel | Why |
|---|---|
| ITE Metropolitan Section (NY) | Transportation engineers, low barrier, technical audience |
| Sam Schwartz, Kittelson, HNTB, Arup | Analysts who produce these deliverables for city clients |
| Transportation Alternatives / Streetsblog NYC | Advocacy side; often knows funding mechanics well and is very reachable |
| NYC DOT borough Vision Zero workshops | DOT staff present and taking questions in person |
| LinkedIn: "Vision Zero analyst," "traffic safety data," "transportation planner NYC" | Direct, and your finding gives the message a real hook |
| NYU Rudin Center, Hunter urban policy | Researchers respond well to a specific methodological question |
