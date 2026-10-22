"use strict";

// Plain-English labels for the five decider questions, in policy priority order.
const QUESTIONS = [
  ["addressed", "Talking to Kuuki"],
  ["contradicts", "Contradicts the reference facts"],
  ["unanswered", "Question left unanswered"],
  ["closing", "Moving on with points open"],
  ["drifted", "Drifted off the agenda"],
];
const INTENT_LABEL = {
  fact_check: "Fact check",
  answer_question: "Answer a question",
  recap: "Recap",
  nudge_back: "Back to the agenda",
  stay_silent: "Stay quiet",
};

const $ = (id) => document.getElementById(id);
let source = null;
let paused = false;
let stats;

function resetStats() {
  stats = { lines: 0, spoke: 0, gateLlm: 0, speakerLlm: 0, latencies: [] };
  renderStats();
}

function renderStats() {
  $("lines").textContent = stats.lines;
  $("spoke").textContent = stats.spoke;
  $("llm").textContent = stats.gateLlm + stats.speakerLlm;
  $("llm-every").textContent = stats.lines + stats.speakerLlm;
  const l = [...stats.latencies].sort((a, b) => a - b);
  $("median").textContent = l.length ? Math.round(l.length % 2 ? l[(l.length - 1) / 2] : (l[l.length / 2 - 1] + l[l.length / 2]) / 2) : "-";
}

function buildBars() {
  $("bars").innerHTML = QUESTIONS.map(([key, label]) => `
    <div class="bar-row" id="row-${key}">
      <div class="bar-label"><span class="name">${label}</span><span class="p" id="p-${key}">-</span></div>
      <div class="track"><div class="fill" id="fill-${key}"></div><div class="threshold" id="t-${key}"></div></div>
    </div>`).join("");
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function addMessage(who, text, ai, intent) {
  const chat = $("chat");
  const m = el("div", "msg" + (ai ? " ai" : ""));
  m.appendChild(el("div", "who", who));
  m.appendChild(el("div", "bubble", text));
  if (intent) {
    m.appendChild(el("br"));
    m.appendChild(el("span", "intent", INTENT_LABEL[intent] || intent));
  }
  chat.appendChild(m);
  chat.scrollTop = chat.scrollHeight;
  return m;
}

function addSys(text, cls) {
  const chat = $("chat");
  chat.appendChild(el("div", "sys" + (cls ? " " + cls : ""), text));
  chat.scrollTop = chat.scrollHeight;
}

let lastLineEl = null;

function showDecision(d) {
  for (const [key] of QUESTIONS) {
    const p = d.probs[key];
    const t = d.thresholds[key];
    const row = $("row-" + key);
    row.classList.toggle("over", p >= t);
    row.classList.toggle("fired", d.trigger === key);
    $("p-" + key).textContent = p.toFixed(2);
    $("fill-" + key).style.width = (p * 100).toFixed(1) + "%";
    const th = $("t-" + key);
    th.style.left = (t * 100).toFixed(1) + "%";
    th.dataset.t = t.toFixed(2);
  }
  const v = $("verdict");
  if (d.interject) {
    v.className = "verdict speak";
    v.textContent = "Speak: " + (INTENT_LABEL[d.intent] || d.intent);
  } else {
    v.className = "verdict quiet";
    v.textContent = "Stay quiet";
    if (lastLineEl) lastLineEl.querySelector(".who").appendChild(el("span", "quiet-mark", "Kuuki stayed quiet"));
  }
  $("reason").textContent = d.reason;
  $("ms").textContent = Math.round(d.latency_ms);
}

function handle(e) {
  switch (e.type) {
    case "start":
      $("meeting-title").textContent = e.title;
      $("status").textContent = "running";
      break;
    case "warmup":
      addSys(`Decider warmed up (${Math.round(e.latency_ms)} ms)`);
      break;
    case "line":
      lastLineEl = addMessage(e.speaker, e.text, false);
      $("latest-line").textContent = `${e.speaker}: "${e.text}"`;
      $("verdict").className = "verdict idle";
      $("verdict").textContent = "Thinking...";
      break;
    case "decision":
      stats.lines += 1;
      stats.gateLlm += e.llm_calls;
      stats.latencies.push(e.latency_ms);
      showDecision(e);
      renderStats();
      break;
    case "speech":
      stats.spoke += 1;
      stats.speakerLlm += e.model_calls;
      addMessage("Kuuki", e.text, true, e.intent);
      renderStats();
      break;
    case "paused":
      addSys("Paused");
      $("status").textContent = "paused";
      break;
    case "resumed":
      addSys("Resumed");
      $("status").textContent = "running";
      break;
    case "summary":
      $("status").textContent = "finished";
      addSys("Meeting finished");
      finish();
      break;
    case "stopped":
      $("status").textContent = "stopped";
      break;
    case "error":
      addSys(e.message, "error");
      $("status").textContent = "error";
      finish();
      break;
  }
}

function finish() {
  $("pause").disabled = true;
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function start() {
  if (source) source.close();
  $("chat").innerHTML = "";
  resetStats();
  buildBars();
  $("verdict").className = "verdict idle";
  $("verdict").textContent = "Starting...";
  $("reason").textContent = "";
  $("latest-line").textContent = "";
  paused = false;
  $("pause").textContent = "Pause";
  try {
    await post("/api/start", { script: $("script").value, speak: $("voice").checked, realtime: true });
  } catch (err) {
    addSys(String(err), "error");
    return;
  }
  $("pause").disabled = false;
  source = new EventSource("/api/events");
  source.onmessage = (m) => handle(JSON.parse(m.data));
  source.addEventListener("end", () => source.close());
}

async function togglePause() {
  const r = await post(paused ? "/api/resume" : "/api/pause");
  paused = r.paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
}

async function init() {
  const names = await (await fetch("/api/scripts")).json();
  $("script").innerHTML = names.map((n) => `<option value="${n}">${n}</option>`).join("");
  buildBars();
  resetStats();
  $("start").onclick = start;
  $("pause").onclick = togglePause;
  $("voice").onchange = () => post("/api/speak", { on: $("voice").checked }).catch(() => {});
}

init();
