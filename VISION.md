# Sequence

## What Sequence is

Sequence is a word-chain game. Each puzzle gives two words — a start and an
end, e.g. `rain → business` — and the player's job is to connect them with a
chain of words where each step makes sense.

There is no answer key. Any chain, of any length, is valid as long as every
adjacent pair of words in it is a real, verifiable link. The game rewards
finding the *shortest* such chain, not guessing a pre-written path.

## How verification works

Every link in a chain — word A to word B — is checked by computing the
cosine similarity between their word2vec embeddings (pretrained
word2vec-google-news-300, trained on real-world text, so it captures things
like `star ↔ wars` from pop culture, not just dictionary synonyms). If the
similarity clears a calibrated threshold, the link is verified.

A full sequence is valid when every adjacent link in it is verified. The
verification engine builds a solution tree for each accepted sequence,
recording the similarity score behind every link, so nothing is a black box
— you can see exactly why each step was accepted.

This lives today as a Python library in `algorithm/`, with four core
operations: `verify_sequence`, `shortest_route`, `build_solution_tree`, and
`is_valid_puzzle`.

## The weekly game loop

Each week we publish one puzzle: a "sequence of the week" start/end pair.

Players submit their own chain, freely. The verification engine checks it
link by link and either signs off on the whole thing or rejects it with a
reason — which link broke, and why. There's no penalty for taking a scenic
route: creative detours are legal. The only requirement is that every link
in the chain is real.

Submissions are ranked on a leaderboard by length — fewest intermediate
words wins. A 3-word chain beats a 7-word chain even if the 7-word chain is
more interesting to read.

## Leaderboard & bounties

Before we publish a puzzle, we run our own solver — a breadth-first search
over a cleaned vocabulary of roughly 50,000 words — and find the shortest
route we can. That route, with its per-link similarity values, is published
as the benchmark players are trying to beat.

If a player finds a shorter verified route than our benchmark, they claim
the bounty. This only works because verification is deterministic and
public: anyone can re-run the same model, the same threshold, and the same
two words, and get the same answer. The engine is the trust anchor, not our
say-so.

## Fairness & trust

A few rules keep the game honest and keep shortest routes interesting
rather than degenerate:

- Connector words must come from the same cleaned vocabulary the solver
  uses — no obscure or malformed tokens that only exist to game a link.
- Trivial inflection chains don't count as real steps. `rain → rains →
  raining` is not a 3-word sequence; it's one idea wearing three outfits.
- Generic hub words — words that sit near almost everything in embedding
  space and can bridge nearly any two concepts — are excluded, so a
  shortest route has to be a genuine discovery, not a trick everyone
  already knows.

## Where the project is now, and what's next

The verification engine exists and works: `verify_sequence`,
`shortest_route`, `build_solution_tree`, and `is_valid_puzzle` are all
implemented in `algorithm/`. An earlier browser prototype was removed once
it had served its purpose of proving the concept.

What's left is the part players will actually touch: a player-facing client
for submitting and exploring sequences, and a leaderboard backend to store
puzzles, submissions, and bounty claims. The hard problem — deciding
whether a link is real — is solved. The remaining work is building the game
around it.
