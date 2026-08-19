# What This Project Is

A plain-language explanation of `python-dpo` — what it does, the thinking behind it, and
where it honestly stands. No mathematics.

For *how* each stage is built, see the numbered reports (`02_PROBLEM_DATASET.md` through
`12_PIPELINE_ORCHESTRATION_AND_PRODUCTIONIZATION.md`). For usage, see `README.md`. This file
is the one to read first.

---

## What it does, in one sentence

It tries to make a small AI coding model better at writing Python — by having the model
write code, running that code against real tests to see what actually works, and then
teaching the model from the difference between its own good and bad attempts.

---

## The problem it's addressing

An AI model that writes code learned to do it by reading enormous amounts of text. It
learned what code *looks like*. Nobody ever checked whether the code it writes actually
**runs**.

That's the gap. The model is optimizing for "this looks like plausible Python," not "this
works." Those are different targets, and the difference is where bugs live.

So how do you teach it the second thing?

---

## The idea

The obvious approach is: hire experts to write correct solutions, and have the model imitate
them. That's expensive, slow, and caps the model at the experts' level.

This project does something else:

1. Give the model a problem
2. Let it write **several different attempts** — five, say
3. **Actually run** each attempt against tests
4. Now you know, objectively, which attempts worked and which didn't
5. Teach the model: *"attempt 3 was better than attempt 1"*

The model becomes its own teacher. The only thing supplied from outside is the problems and
the tests. Nobody has to write correct answers.

**The crucial move is step 5 — teaching by comparison rather than correction.** You're not
saying "here's the right answer." You're saying "between these two things you produced, this
one was better." That turns out to be enough for a model to shift its behavior, and it's far
cheaper to obtain, because a test result gives you the comparison for free.

That technique has a name — DPO, Direct Preference Optimization — but the name doesn't
matter much. The idea is: learn from pairs of your own attempts, ranked by something
objective.

---

## The philosophy

This is the part that explains why the codebase looks the way it does. A few convictions are
baked into it.

### 1. Truth comes from execution, not opinion

"Which attempt was better" is never decided by a human rater or by another AI grading the
work. It's decided by running the code. A test passes or it doesn't. This is the
foundational commitment, and much of the design follows from refusing to compromise it.

### 2. The main danger is fooling yourself

This is the conviction that shapes the most code, and it deserves the most space.

When you measure someone else's work, a mistake gives you a wrong answer. When you measure
**your own** work, a mistake tends to give you a *flattering* wrong answer — because errors
that make results look good feel like success and don't prompt investigation, while errors
that make results look bad feel like bugs and get hunted down.

That asymmetry is the whole problem. Left alone, a pipeline drifts toward good-looking
numbers, and nobody notices, because at every step the person checking is the person hoping.

So the system is full of **refusals**. Each one blocks a specific, named way this kind of
project lies to itself.

#### Measuring something you already taught it

*The mistake:* train on a problem, then test on that same problem. The model has effectively
seen the answer sheet. The score goes up and means nothing.

*Why it's insidious:* it is invisible downstream. The benchmark file is content-hashed and
looks pristine — the contamination lives upstream, in the training data. Nothing about the
evaluation report would look wrong. You would get a genuinely better number and a genuinely
worthless one, with no way to tell them apart later.

*The guard:* before every evaluation, the benchmark is cross-referenced against the training
split, and any overlap stops the run. This has fired in practice:

```
Stage model_evaluation failed: benchmark 'python_eval_v1' overlaps
the preference train split: p005
```

That run had already spent eleven minutes generating and training. It discarded the result
rather than report a contaminated number.

A second copy of the guard sits in the analysis stage, deliberately harsher: when a refined
dataset would carry a held-out problem, it **raises an error rather than filtering the row
out**. Filtering would let a leak be introduced and quietly corrected, leaving no trace it
was attempted. An exception makes the attempt visible.

#### Measuring nothing and calling it a result

*The mistake:* run training with an empty or near-empty dataset. You get an adapter. It is
labelled "DPO-trained." It is really the base model with noise on top.

*Why it's insidious:* everything downstream works perfectly. Evaluation runs, produces
numbers, shows no improvement — and the conclusion drawn is *"DPO doesn't help this model,"*
when the real conclusion was *"nothing was ever trained."* Those look identical from outside.

*The guard:* training refuses to start on an empty split. This has fired in practice:

```
Stage dpo_training failed: the training split is empty;
there is nothing to train on
```

The companion guard refuses to train without a validation split, rather than training blind
and reporting a loss curve nobody can interpret.

#### Reading noise as signal

*The mistake:* draw a conclusion from a sample too small to support one.

*Why it's insidious:* small samples produce *large* apparent effects. With seven benchmark
problems, a single problem flipping moves the headline number by fourteen percentage points
— a dramatic-looking result generated entirely by chance.

*The guard:* the analysis stage checks an evidence floor *before* any other verdict. Below
thirty benchmark problems, or with a result range too wide to be informative, the answer is
`insufficient_evidence`, and no other conclusion may be reported as the headline however
suggestive it looks. The report says so in its opening paragraph rather than burying it:

> This decision gates every other finding below. The analyses still ran and their numbers
> are reported, but the evidence does not meet the configured minimum, so none of them is
> offered as a conclusion about model quality.

On the real data, the coverage and failure analyses both found things worth acting on. The
gate suppressed them anyway. That is the guard working against its own findings.

#### Moving the goalposts

*The mistake:* run the experiment, see the result, then decide what counts as success.

*Why it's insidious:* it never feels like cheating. It feels like refining your criteria in
light of what you learned.

*The guard:* success criteria live in the configuration file, fixed before the run. The
evaluation reports each clause separately and then the overall verdict, so a `false` is
legible rather than a single number open to reinterpretation.

The strongest evidence this holds is that the project published a negative verdict on its
own model:

```
pass_at_1_improves: False
DPO_SUCCESS: False
```

Adjusting the thresholds afterwards to produce a pass would have taken about one line. It
was not done.

#### Losing the inconvenient data

*The mistake:* quietly drop what does not fit — failed generations, rejected pairs,
candidates that crashed. The dataset gets cleaner and the statistics get wrong, because the
attrition is invisible.

*The guard:* every generated candidate, evaluation failure and rejected preference pair is
persisted **with its rejection reason**. The refinement plan records a verdict for every
pair, including the ones it keeps, so it is a complete audit rather than a list of
survivors.

Empty files are written deliberately for the same reason. If there were no regressions, the
regressions file exists and is empty — so "no regressions" stays distinguishable from "the
stage never ran."

#### Dressing up coincidence as cause

*The mistake:* observe that a weakly-covered category also has failures, and write "DPO
failed because of insufficient data in that category."

*Why it's insidious:* it is a perfectly reasonable hypothesis. It is simply not something
the observation establishes — least of all here, where each category holds a single problem.

*The guard:* reports say *"potential data gap"* and never use a causal verb. This is not a
style preference: there is a list of forbidden phrases and a test asserting none of them
appears in the rendered output. Every analysis report also ends with a section titled "What
this analysis does not establish."

A related rule: the report names a likely failure mode only when a specific error type
actually dominates. With no clear pattern the line is **omitted** rather than filled with a
guess.

#### Trusting a judgement you cannot check

*The mistake:* use a language model to grade the outputs.

*Why it's insidious:* it is convenient, it scales, and it produces plausible labels. But
those labels cannot be audited or reproduced, and everything built on them inherits that.
The recommendations become unfalsifiable.

*The guard:* classification is deterministic — test status, error type, exit code, timeout
flag. An AI judge is explicitly forbidden as the primary classifier. Same inputs, same
label, every time, and anyone can check it by hand.

#### Promoting on hope

*The mistake:* a model looks good, so it quietly becomes the default.

*The guard:* packaging registers models as `EXPERIMENTAL` only. Reaching `RECOMMENDED`
requires passing through `VALIDATED` first — so nothing is recommended sight-unseen — plus a
recorded passing evaluation, which the tool reads from the report rather than accepting on
assertion.

#### The principle underneath all of it

These guards share something: they are enforced **by construction, not by discipline**.

The clearest case is in the analysis code, where a recommendation must carry evidence and a
stated hypothesis — so the object refuses to be constructed without them:

```python
if not isinstance(self.evidence, dict) or not self.evidence:
    raise AnalysisStoreError(
        "recommendation.evidence must be a non-empty mapping (spec section 55): "
        "a recommendation without evidence is an opinion"
    )
```

This could have been a code-review convention instead. It would have held for a while and
then eroded — someone in a hurry, someone new, someone who does not know why the rule
exists. As a constructor check it cannot erode: an unsupported recommendation is
unrepresentable.

That is the pattern throughout. The guards are not reminders to be careful. They are
structures that make the careless thing impossible.

### 3. Untrusted code is untrusted

The model writes code, and that code gets run. It could delete files or hang forever — not
maliciously, just because the model made a mistake. So generated code only ever executes
inside a locked-down container: no network, no access to the host filesystem, a non-root
user, and hard limits on time, memory and processes.

### 4. Everything must be traceable

If a trained model turns out good or bad, you need to know exactly what produced it — which
comparisons trained it, which test results produced those comparisons, which attempts
produced those results, which problems produced those attempts. Every step records its
inputs and outputs with checksums, and the full chain is stored as a fact rather than
reconstructed by hand.

### 5. Nothing gets promoted automatically

A trained model is registered as "experimental" and stays there. Marking it as good requires
a human to do so explicitly, backed by a passing evaluation record.

---

## Why it looks disproportionate

Here is the thing most likely to confuse a newcomer: **the machinery is enormous relative to
what it has produced.** Fifty-odd thousand lines of code, twelve stages, over fifteen hundred
tests — and the model it trained learned essentially nothing, from three examples.

That mismatch is real, and it is not a mistake. The project was built **apparatus first**.
The reasoning: a preference-learning pipeline that quietly produces wrong numbers is worse
than useless, so build the thing that can be trusted, prove it works end to end, and only
then feed it real data.

Think of it as building a laboratory before running the experiment. The lab is finished and
verified. The experiment has not really been run.

---

## How the pieces fit together

The pipeline runs as a sequence, each step consuming what the previous one produced:

**Problems** — a curated set of Python tasks, each with a reference solution and tests.

**Generation** — the model writes several attempts at each problem, using varied prompting
styles to encourage different approaches.

**Execution** — each attempt runs against its problem's tests, inside the sandbox. This is
where objective evidence gets created.

**Ranking** — attempts are scored and ordered by how they actually performed.

**Preference pairs** — for each problem, better attempts are paired against worse ones.
Comparisons are only ever made *within* a single problem; comparing an attempt at one
problem against an attempt at another would be meaningless.

**Training** — the model is fine-tuned on those pairs. Only a small adapter is trained, not
the whole model, which is what makes this feasible on a single consumer graphics card.

**Evaluation** — the original model and the tuned model are compared on held-out problems
neither was trained on, generating and executing code for both under identical conditions.

**Analysis** — failures are classified, gaps between what was trained and what was measured
are identified, and a proposed next experiment is written out. It stops there; it never
retrains on its own.

**Packaging** — the trained adapter is bundled into something loadable, verified by making
it generate code and running that code, then registered.

A single command runs all of it, and one identifier ties every artifact together.

---

## Where it actually stands

The pipeline works. It has been run end to end on real hardware — real code generation, real
sandboxed execution, real training, real packaging.

**But the experiment cannot produce a meaningful result yet, and the system says so rather
than pretending otherwise.**

The reason is the problem set. There are only ten problems. Of those, the model reliably
succeeds at four and reliably fails at three. Those seven are useless for this technique — if
every attempt is equally good, or equally bad, there is nothing to compare. Only three
problems sit in the zone where the model sometimes succeeds and sometimes fails, and one of
those is reserved for testing.

That leaves roughly two usable problems. Not enough to teach anything.

The analysis stage adds a second finding, and it is arguably sharper: the problems that were
trained on and the problems used for evaluation **share no categories at all**. Nothing that
was trained was ever measured, and nothing that was measured was ever trained.

**The fix is not more code.** It is more problems — going from ten hand-written ones to a few
hundred, so that enough of them sit where the model is genuinely uncertain, and so the
training and evaluation sets can overlap in subject matter. That is a content-writing effort,
not an engineering one.

---

## The honest summary

What exists is a working, trustworthy apparatus for preference-based training of a code
model, built to refuse rather than flatter, and verified end to end.

What does not exist yet is a result. The dataset is too small to produce one, and the
pipeline reports that plainly instead of manufacturing a number.
