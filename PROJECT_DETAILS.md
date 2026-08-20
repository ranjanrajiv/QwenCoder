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

The model writes code, and that code gets run. Not eventually — running it is the entire
point, because execution is what produces the training signal.

The threat is rarely malice. It is that a model asked to write a sorting function writes an
infinite loop, or allocates until the machine swaps, or writes a file where it shouldn't.
A model told to solve a problem it cannot solve will do *something*, and "something" is
unbounded.

#### The boundary

Every generated candidate runs inside a container built with a fixed argument list:

| Flag | What it removes |
|---|---|
| `--network none` | No network at all. Nothing can phone home, fetch, or exfiltrate. |
| `--cap-drop ALL` | Every Linux capability dropped — no raw sockets, no mounting, no ptrace. |
| `--security-opt no-new-privileges` | A process cannot gain privileges it did not start with. |
| `--user <non-root>` | Not root, even inside the container. |
| `--read-only` + `--tmpfs` | Root filesystem is read-only; the only writable space is a small, bounded tmpfs that vanishes with the container. |
| `--pids-limit` | Fork bombs hit a ceiling. |
| `--cpus`, `--memory`, `--memory-swap` | Bounded CPU and memory, with swap pinned to the memory limit so it cannot be dodged by swapping. |
| a wall-clock timeout | Infinite loops die. |

Only the job's own workspace is mounted, and the working directory is set to it.

#### Two properties that matter more than the flags

**Candidate code never becomes part of a command.** The generated source is written to a
file, and the container command is a fixed argument list referencing that file by path.
`shell=True` appears nowhere in the source. So there is no string interpolation of
model-written text into anything that gets executed as a command — the classic injection
route simply does not exist.

**Generated tests are treated as untrusted too.** This is the non-obvious one. The pipeline
builds a pytest file around each candidate. That file's literals come from the problem's
already-validated test data, never from the candidate — but it *runs in the same process as
the candidate's code*. So the bundle is confined exactly like a bare candidate would be.
Trusting the test file because "we wrote it" would hand the candidate a way out.

#### The guarantees are tested, not asserted

`tests/sandbox/test_sandbox_security.py` demonstrates each property against a real Docker
daemon rather than documenting it: network disabled, non-root, capabilities dropped,
privilege escalation blocked, read-only root with a bounded tmpfs, resource limits actually
enforced, swap pinned, only the workspace mounted, the command a fixed argument list, and a
parametrised test asserting that a list of dangerous flags never appears.

#### The one deliberate exception

**Reference solutions** — the trusted answers shipped with the problem catalog — execute
in-process, without a container. They are written by hand, reviewed like any other source
file, and live in the repository. That exception is confined behind a protocol boundary
(`InProcessReferenceExecutor`), so model-written code cannot reach it. It is also the thing
that would have to change first if the catalog ever grew by importing problems from
elsewhere: a thousand third-party reference solutions could not honestly be called reviewed.

Inspecting a candidate is different from running one. The pipeline parses generated code
into a syntax tree to check whether it is valid Python and whether it defines the expected
function. Building a syntax tree does not import, evaluate, or execute anything, which is
why that check is allowed to touch untrusted code directly while everything else is not.

### 4. Everything must be traceable

Six months after a training run, someone asks a simple question: *what exactly produced this
model?* Without an answer recorded at the time, you are reconstructing from memory and
directory timestamps — and memory reliably favours the version of events that explains the
result you got.

#### The chain, recorded rather than reconstructed

Every trained adapter traces back through the full pipeline. From a real committed run:

```
model_adapter
  ├── training_run_id       dpo_20260819_061731_8314
  ├── preference_run_id     pref_20260819_061728_849c
  ├── ranking_run_id        rank_20260819_061722_9de4
  ├── evaluation_run_id     eval_20260819_061432_5204
  ├── candidate_run_id      run_20260819_060953_1cba
  └── problem_dataset_run_id …
```

Before this existed, that chain was implicit — you could follow it by opening one manifest,
reading an id, opening the next. Nothing recorded it, nothing verified it, and nothing would
have noticed if a dataset were regenerated underneath the adapter. It is now written to
`lineage.json` as a fact.

#### Content, not just names

Names are not enough, because a file can change while keeping its name. So artifacts are
identified by **SHA-256 of their contents**:

- Each candidate's code is hashed, which is also how duplicate attempts are detected.
- Each training split is hashed, and the hashes are recorded on the training manifest — so
  "the data this model trained on" is a checkable claim rather than a recollection.
- The benchmark manifest carries a hash of the problems it selects, so silent drift is
  detectable.
- The experiment's `artifacts.json` records every stage output by path, hash and size.

This is also what makes the cache trustworthy. A stage is reused only when its inputs' *content
hashes* match — not when its name matches, and not on a timestamp.

#### Writes that survive a crash

Reproducibility is worthless if an interrupted run leaves a half-written file that looks
complete. Every JSON write goes: write to a temporary file → `fsync` it → `os.replace` into
position → `fsync` the directory. `os.replace` is atomic, so a reader sees either the old
file or the new one, never a partial one. Append-only logs fsync each line, and the reader
treats a torn final line as an error rather than skipping it.

That is what makes resume safe: a run can be interrupted at any point and continued, because
what is on disk is always a consistent prefix of what was intended.

#### Recording what could not be reproduced

Traceability includes being honest about its own limits. Each run captures the environment
it ran in — Python, CUDA, GPU, library versions, Docker — and the git commit, along with
whether the working tree was **dirty** at the time. A result produced from uncommitted
changes is flagged as such rather than silently attributed to the last commit.

Hostnames and usernames are deliberately never captured. Provenance should identify the
software and the data, not the person.

### 5. Nothing gets promoted automatically

The failure this prevents is quiet and common: a model scores well, someone points something
at it "just to try," and months later it is load-bearing. Nobody decided that. It accumulated.

#### Packaging can only ever register one status

When a trained adapter is packaged, it enters the registry as `EXPERIMENTAL`. Not
"probably fine," not "good enough" — the packaging code is *incapable* of writing any other
status. Registering anything higher requires a separate, explicit command.

#### The path upward has no shortcut

```
EXPERIMENTAL ──> VALIDATED ──> RECOMMENDED ──> RETIRED
     │                │
     └──> RETIRED     └──> RETIRED
     └──> REJECTED    └──> REJECTED
```

`RECOMMENDED` is reachable **only through** `VALIDATED`. There is no edge from
`EXPERIMENTAL` straight to `RECOMMENDED`, so a model cannot be recommended without someone
having first looked at it and said so. `RETIRED` and `REJECTED` are terminal — an
un-retirement is a new entry, not a quiet reversal.

#### And the top step demands evidence

Promoting to `RECOMMENDED` requires an evaluation run id *and* a passing success-criteria
record. Supply neither and it refuses:

```
cannot promote 'exp_...' to RECOMMENDED without an evaluation_run_id

cannot promote 'exp_...' to RECOMMENDED: evaluation run 'eval_...'
did not pass its recorded success criteria
```

The command reads the pass/fail verdict out of the evaluation report on disk rather than
accepting the caller's assertion that the model is good. The person promoting cannot simply
believe it.

#### Verification is a precondition, not a checkbox

Before anything reaches the registry at all, the packaged model must prove it works: it is
loaded, asked to write a function, and **the code it produces is executed through the
sandbox**. Only if those tests pass is the model registered.

There is no flag to skip this. A verification failure raises before any registry entry is
written, so an unverified package cannot exist in the registry even in a broken state.

#### Why this belongs in a philosophy section

Every other guard here protects a *number* from being wrong. This one protects a *decision*
from being made by default.

The connection to point 2 is direct: automatic promotion is the purest form of fooling
yourself, because nobody ever consciously concluded anything. Requiring an explicit act —
with evidence the tool checks itself — means there is always a person and a record behind
the claim that a model is good.

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
