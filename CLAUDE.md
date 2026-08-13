# nyc-crash-risk-forecast

Rebuilds NYC DOT's Vision Zero pedestrian-safety priority ranking using the Highway Safety
Manual standard — a negative-binomial Safety Performance Function blended with observed
counts via Empirical Bayes — and backtests it against held-out years.

See [README.md](README.md) for the result and its caveats, [TODOS.md](TODOS.md) for what is
deferred and why.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
