const elements = {
  chain: document.querySelector("#chain"),
  wordForm: document.querySelector("#word-form"),
  wordInput: document.querySelector("#word-input"),
  submitWord: document.querySelector("#submit-word"),
  gameStatus: document.querySelector("#game-status"),
  nextPair: document.querySelector("#next-pair"),
  helpToggle: document.querySelector("#help-toggle"),
  helpPanel: document.querySelector("#help-panel"),
};

const unlikelyPairIds = new Set([
  "stone:city",
  "winter:guitar",
  "wizard:subway",
  "pizza:library",
]);

const reasonMessages = {
  "below-threshold": "That connection is too weak.",
  "not-in-vocabulary": "The engine does not recognize that word.",
  "not-in-pool": "Try a more common connector word.",
  "inflection-variant": "A simple variation of the same word does not count.",
  "repeated-word": "That word is already in the chain.",
};

let puzzles = [];
let currentPuzzle = null;
let acceptedWords = [];
let complete = false;

function setStatus(message, state = "neutral") {
  elements.gameStatus.textContent = message;
  elements.gameStatus.classList.toggle("is-error", state === "error");
  elements.gameStatus.classList.toggle("is-complete", state === "complete");
}

function setFormDisabled(disabled) {
  elements.wordInput.disabled = disabled;
  elements.submitWord.disabled = disabled;
}

function setHelpOpen(open) {
  elements.helpPanel.hidden = !open;
  elements.helpToggle.setAttribute("aria-expanded", String(open));
}

function renderChain() {
  if (!currentPuzzle) {
    elements.chain.replaceChildren();
    return;
  }

  const words = [currentPuzzle.start, ...acceptedWords, currentPuzzle.end];
  const items = words.map((word, index) => {
    const item = document.createElement("li");
    const value = document.createElement("span");
    item.className = "chain-word";
    value.className = "chain-value";
    value.textContent = word;

    if (index === 0 || index === words.length - 1) {
      const label = document.createElement("span");
      label.className = "chain-label";
      label.textContent = index === 0 ? "Start" : "Target";
      item.classList.add(index === 0 ? "chain-start" : "chain-end");
      item.append(label);
    }

    item.append(value);
    return item;
  });

  elements.chain.replaceChildren(...items);
  elements.chain.classList.toggle("is-complete", complete);
  elements.chain.setAttribute("aria-label", `Current chain: ${words.join(" to ")}`);
}

function resetCurrentPair() {
  acceptedWords = [];
  complete = false;
  elements.wordInput.value = "";
  elements.nextPair.hidden = true;
  setFormDisabled(false);
  renderChain();
  setStatus(`Enter a word that connects to “${currentPuzzle.start}”.`);
  elements.wordInput.focus();
}

function chooseNewPair() {
  if (puzzles.length === 0) {
    return;
  }

  const choices = puzzles.filter((puzzle) => puzzle !== currentPuzzle);
  const pool = choices.length > 0 ? choices : puzzles;
  currentPuzzle = pool[Math.floor(Math.random() * pool.length)];
  resetCurrentPair();
}

async function requestVerification(words) {
  const response = await fetch("/api/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      start: currentPuzzle.start,
      end: currentPuzzle.end,
      words,
    }),
  });

  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`The server returned an unreadable response (${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(data.error?.message || `The request failed (${response.status}).`);
  }
  return data;
}

async function submitNextWord(event) {
  event.preventDefault();
  if (!currentPuzzle || complete) {
    return;
  }

  const candidate = elements.wordInput.value.trim();
  if (!candidate) {
    setStatus("Enter a word before submitting.", "error");
    elements.wordInput.focus();
    return;
  }

  const previous = acceptedWords.at(-1) || currentPuzzle.start;
  const attemptedWords = [...acceptedWords, candidate];
  setFormDisabled(true);
  setStatus(`Checking “${previous}” → “${candidate}”…`);

  try {
    const result = await requestVerification(attemptedWords);
    const newLink = result.links[attemptedWords.length - 1];

    if (!newLink?.verified) {
      const explanation = reasonMessages[newLink?.reason]
        || "Those words do not form a verified connection.";
      setStatus(
        `“${candidate}” does not connect to “${previous}”. ${explanation}`,
        "error",
      );
      elements.wordInput.select();
      return;
    }

    acceptedWords.push(candidate);
    elements.wordInput.value = "";

    if (result.verified) {
      complete = true;
      renderChain();
      setStatus("Verified chain.", "complete");
      elements.nextPair.hidden = false;
      return;
    }

    renderChain();
    setStatus(`Keep going toward “${currentPuzzle.end}”.`);
    elements.wordInput.focus();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setFormDisabled(complete);
  }
}

async function initialize() {
  setFormDisabled(true);
  try {
    const response = await fetch("/api/puzzles", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Could not load the preset words (${response.status}).`);
    }
    const data = await response.json();
    const unlikelyPuzzles = data.puzzles.filter((puzzle) =>
      unlikelyPairIds.has(`${puzzle.start}:${puzzle.end}`),
    );
    puzzles = unlikelyPuzzles.length > 0 ? unlikelyPuzzles : data.puzzles;
    chooseNewPair();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

elements.wordForm.addEventListener("submit", submitNextWord);
elements.nextPair.addEventListener("click", chooseNewPair);
elements.helpToggle.addEventListener("click", () => {
  setHelpOpen(elements.helpToggle.getAttribute("aria-expanded") !== "true");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setHelpOpen(false);
    elements.helpToggle.focus();
  }
});
document.addEventListener("click", (event) => {
  if (
    elements.helpToggle.getAttribute("aria-expanded") === "true"
    && !elements.helpPanel.contains(event.target)
    && event.target !== elements.helpToggle
  ) {
    setHelpOpen(false);
  }
});

initialize();
