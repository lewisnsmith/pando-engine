# Game and scoring ideas

This document records behavior excluded from the generalized connectivity
engine. None of these ideas changes link or sequence verification until it has
a data source, evaluation method, versioned rule, and explicit product use.

## Generalization requirement

Runtime verification must apply the same rule to every input pair. The engine
must not contain pair allowlists, named exceptions, seed puzzles, or special
handling for words listed in the repository. Tests may use example tokens, but
test data must not affect production acceptance.

Any future extension must satisfy these requirements:

- Evaluate unseen word pairs without repository changes.
- Define symmetric or directional behavior explicitly.
- Identify every model, dataset, configuration value, and generated artifact.
- Measure false acceptance and false rejection on held-out human judgments.
- Preserve the generalized cosine result alongside any additional score.

## Thematic puzzles

A theme can influence puzzle selection without changing sequence validity. An
editorial process can select start and target words from a theme, then use the
general engine to validate every submitted sequence.

Thematic weighting would add a separate score. One possible definition assigns
the theme an embedding vector and measures each connector against that vector.
This score should remain independent from the binary link predicate so a valid
sequence does not depend on an unpublished theme rule.

Questions for evaluation include:

- Whether the theme score rewards coherent routes or repeated synonyms.
- Whether players can understand the score from the published rules.
- Whether weighting changes the shortest-route objective.
- Whether a theme uses authored labels, embedding centroids, or another model.

## Entities and cultural references

The removed collocation rule accepted a pair when Word2Vec contained an
underscore-joined token such as `Star_Wars`. This rule applied inconsistently:
player verification accepted collocation edges while breadth-first search used
cosine-threshold edges for most of its search.

Possible generalized approaches include:

- Use phrase and entity embeddings from a versioned model.
- Add edges from a versioned knowledge graph with a published relation policy.
- Train a pair classifier from human link judgments.
- Combine cosine and entity evidence through a calibrated score rather than a
  named-pair exception.

Evaluation should include cultural references absent from the development set,
ordinary multiword entities, ambiguous names, and unrelated pairs that happen
to form a model token.

## Model replacement and collocation research

The Google News Word2Vec artifact should remain a versioned baseline while
candidate models are evaluated. Replacing it changes every cosine, pool edge,
hub degree, route, and acceptance threshold. A replacement therefore requires
a new model identifier, connector pool, calibration result, and compatibility
decision.

Three model tracks merit comparison:

1. Train a new static Word2Vec model on a dated, licensed corpus. Include a
   current Wikipedia export and other sources that cover contemporary language.
   Record corpus sources, snapshot dates, preprocessing, hyperparameters,
   random seeds, software versions, and artifact hashes.
2. Evaluate fastText as an alternative static model. Its subword vectors can
   represent vocabulary items absent from the training corpus, but spelling
   similarity may produce links that conflict with the game rule. Evaluate
   semantic quality, proper nouns, misspellings, and inflection behavior
   separately.
3. Train a phrase-aware model. Detect multiword expressions before Word2Vec
   training so the corpus contains stable phrase tokens. Phrase detection must
   use a general corpus statistic, such as normalized pointwise mutual
   information, rather than a repository list of accepted names.

A phrase token alone does not define the connection between its component
words. Collocation handling should retain two independent measurements:

$$
  \operatorname{semantic}(a,b)=\cos(v_a,v_b)
$$

and

$$
  \operatorname{association}(a,b)=\operatorname{NPMI}(a,b).
$$

Cosine estimates distributional similarity. Normalized pointwise mutual
information estimates whether the words occur together more often than their
individual frequencies predict. A calibrated classifier or explicit formula
could combine both measurements after evaluation against held-out human link
judgments. The engine should return both component scores so acceptance remains
inspectable.

The first experiment should compare these candidates against the existing
engine on one frozen evaluation set:

- The existing Google News Word2Vec model.
- A newly trained Word2Vec model without phrase detection.
- The same corpus and settings with phrase detection.
- A fastText model trained on the same corpus.
- A two-signal scorer using cosine and normalized pointwise mutual information.

The evaluation set should separate semantic similarity, direct association,
named entities, cultural references, unrelated pairs, morphology, spelling
similarity, and words absent from one or more model vocabularies. Model
selection should use held-out results rather than examples used to choose the
method.

Primary references:

- [Distributed Representations of Words and Phrases and Their
  Compositionality](https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)
  defines phrase-aware Skip-gram training.
- [Gensim phrase detection](https://radimrehurek.com/gensim/models/phrases.html)
  supports count-based and normalized pointwise mutual information scoring.
- [fastText word representations](https://fasttext.cc/docs/en/unsupervised-tutorial.html)
  describes subword-based static embeddings and out-of-vocabulary vectors.
- [Wikimedia data exports](https://dumps.wikimedia.org/) provide dated public
  corpus snapshots.

## Connector policies

The core engine currently treats inflection rejection, stop-word exclusion,
and high-degree hub suppression as general game rules. Each policy should
remain only while measured tests show that it improves route quality across
unseen words.

Alternative policies include:

- Replace suffix stripping with a versioned morphological analyzer.
- Publish the canonical connector pool as a hashed artifact.
- Replace percentile-based hub removal with a documented degree range.
- Report policy rejection separately from semantic similarity.

## Benchmark and route behavior

The core engine needs one minimum-connector benchmark. The following behaviors
remain optional product or research features:

- Enumerating multiple shortest routes.
- Ranking tied routes by total cosine.
- Revealing a benchmark after a puzzle closes.
- Publishing a commitment hash before play begins.
- Providing route-derived hints.
- Identifying common or unusual community links.

Any benchmark must use the same legal-edge predicate as player verification or
state the difference explicitly.

## Product-state behavior

Authentication, drafts, submissions, personal bests, publication schedules,
and leaderboards belong outside the generalized engine.

A live leaderboard should expose only:

- Rank.
- Player identity or display name.
- Connector count.
- Tie and achievement-time metadata.

It should keep connector words private while the puzzle is active. A separate
reveal policy can publish selected routes after the puzzle closes.

## Experimental process

Evaluate an idea outside the core engine in this order:

1. Define the rule and required artifacts.
2. Assemble training and held-out evaluation pairs.
3. Measure behavior against the cosine-only engine.
4. Document versioning and reproducibility requirements.
5. Decide whether the result belongs in puzzle selection, optional scoring, or
   link verification.
6. Add the behavior through a general interface without named word cases.
