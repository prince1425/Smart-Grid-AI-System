/* -------------------------------------------------------------------------
   Smart Grid AI - Front-end logic
   ------------------------------------------------------------------------- */

const API = {
    energy:     "/get-energy-data?hours=48",
    predict:    "/predict-load?horizon=24",
    fault:      "/detect-fault?hours=48",
    historical: "/historical-data?hours=168",
    optimize:   "/optimize",
};

const AUTO_REFRESH_MS = 10_000;
let demandChart = null;
let anomalyChart = null;
let simTimer = null;

/* ------------------------------------------------------------------ */
/* Utility                                                            */
/* ------------------------------------------------------------------ */
async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} returned ${r.status}`);
    return r.json();
}

function fmt(n, digits = 0) {
    if (n === null || n === undefined || Number.isNaN(n)) return "--";
    return Number(n).toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    });
}

function addAlert(message, level = "info") {
    const feed = document.getElementById("alertFeed");
    if (feed.querySelector(".muted")) feed.innerHTML = "";
    const li = document.createElement("li");
    li.className = level;
    const ts = new Date().toLocaleTimeString();
    li.innerHTML = `<strong>[${ts}]</strong> ${message}`;
    feed.prepend(li);
    while (feed.children.length > 15) feed.removeChild(feed.lastChild);
}

/* ------------------------------------------------------------------ */
/* Charts                                                             */
/* ------------------------------------------------------------------ */
function initCharts() {
    const demandCtx = document.getElementById("demandChart").getContext("2d");
    demandChart = new Chart(demandCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Observed (MWh)",
                    data: [],
                    borderColor: "#4cc9f0",
                    backgroundColor: "rgba(76,201,240,0.15)",
                    fill: true,
                    tension: 0.35,
                    pointRadius: 0,
                    borderWidth: 2,
                    spanGaps: true,
                },
                {
                    label: "Forecast (MWh)",
                    data: [],
                    borderColor: "#b5179e",
                    borderDash: [6, 4],
                    fill: false,
                    tension: 0.35,
                    pointRadius: 0,
                    borderWidth: 2,
                    spanGaps: true,
                },
            ],
        },
        options: chartBaseOptions(),
    });

    const anomalyCtx = document.getElementById("anomalyChart").getContext("2d");
    anomalyChart = new Chart(anomalyCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Demand",
                    data: [],
                    borderColor: "#4cc9f0",
                    backgroundColor: "rgba(76,201,240,0.10)",
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2,
                    spanGaps: true,
                },
                {
                    label: "Anomalies",
                    data: [],
                    borderColor: "#e74c3c",
                    backgroundColor: "#e74c3c",
                    showLine: false,
                    pointRadius: 6,
                    pointStyle: "triangle",
                },
            ],
        },
        options: chartBaseOptions(),
    });

    // Attach to window for debugging
    window.demandChartInstance = demandChart;
    window.anomalyChartInstance = anomalyChart;
}

function chartBaseOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
            legend: {
                labels: { color: "#e6edf7", usePointStyle: true, font: { size: 12 } },
            },
            tooltip: {
                backgroundColor: "#1d2d4d",
                borderColor: "#4cc9f0",
                borderWidth: 1,
            },
        },
        scales: {
            x: {
                ticks: { color: "#8b9bb8", maxTicksLimit: 10 },
                grid:  { color: "rgba(139,155,184,0.08)" },
            },
            y: {
                ticks: { color: "#8b9bb8" },
                grid:  { color: "rgba(139,155,184,0.08)" },
                beginAtZero: false,
            },
        },
    };
}

/* ------------------------------------------------------------------ */
/* Data loaders                                                       */
/* ------------------------------------------------------------------ */
async function loadEnergy() {
    const res = await fetchJSON(API.energy);
    if (!res.ok) throw new Error(res.error || "energy fetch failed");

    const rows = res.data;
    const latest = rows[rows.length - 1] || {};
    document.getElementById("currentValue").textContent = fmt(latest.value, 0);
    document.getElementById("sourceLabel").textContent = res.source;
    document.getElementById("lastUpdated").textContent =
        "Updated: " + new Date().toLocaleTimeString();

    const badge = document.getElementById("statusBadge");
    badge.textContent = latest.status || "Normal";
    badge.className = "status";
    if (latest.status === "Overload") badge.classList.add("overload");
    else if (latest.status === "Under-utilized") badge.classList.add("under");

    // Feed observed series into both charts
    const labels = rows.map(r => r.timestamp.slice(5, 16).replace("T", " "));
    const values = rows.map(r => r.value);

    demandChart.data.labels = [...labels];
    demandChart.data.datasets[0].data = [...values];
    demandChart.data.datasets[1].data = new Array(values.length).fill(null);
    demandChart.update();

    if (latest.status === "Overload") {
        addAlert(`Overload detected at ${labels[labels.length - 1]} (${fmt(latest.value)} MWh)`, "danger");
    }
}

async function loadForecast() {
    const res = await fetchJSON(API.predict);
    if (!res.ok) throw new Error(res.error || "forecast failed");

    const f = res.forecast;
    const fLabels = f.map(r => r.timestamp.slice(5, 16).replace("T", " "));
    const fValues = f.map(r => r.predicted_value);

    // Append forecast to the demand chart after the observed data
    const prevLabels = demandChart.data.labels;
    const prevValues = demandChart.data.datasets[0].data;

    demandChart.data.labels = [...prevLabels, ...fLabels];
    demandChart.data.datasets[0].data = [...prevValues, ...new Array(fLabels.length).fill(null)];
    demandChart.data.datasets[1].data = [
        ...new Array(prevValues.length).fill(null),
        ...fValues,
    ];
    demandChart.update();

    const peak = Math.max(...fValues);
    const peakIdx = fValues.indexOf(peak);
    document.getElementById("peakValue").textContent = fmt(peak, 0);
    document.getElementById("peakTime").textContent = "at " + fLabels[peakIdx];

    const m = res.metrics || {};
    document.getElementById("modelMetrics").textContent =
        `Model R2=${fmt(m.r2, 2)} · MAE=${fmt(m.mae, 1)}`;
}

async function loadFaults() {
    const res = await fetchJSON(API.fault);
    if (!res.ok) throw new Error(res.error || "fault detection failed");

    document.getElementById("anomalyCount").textContent = res.anomaly_count;

    const labels = res.series.map(r => r.timestamp.slice(5, 16).replace("T", " "));
    const values = res.series.map(r => r.value);
    const markers = res.series.map(r => (r.anomaly ? r.value : null));

    anomalyChart.data.labels = labels;
    anomalyChart.data.datasets[0].data = values;
    anomalyChart.data.datasets[1].data = markers;
    anomalyChart.update();

    res.anomalies.slice(-3).forEach(a => {
        addAlert(`Anomaly (${a.severity}) at ${a.timestamp}: ${fmt(a.value)} MWh`,
                 a.severity === "High" ? "danger" : "warn");
    });
}

async function loadSuggestions() {
    const res = await fetchJSON(API.optimize);
    if (!res.ok) throw new Error(res.error || "optimize failed");

    const list = document.getElementById("suggestionList");
    list.innerHTML = "";
    res.suggestions.forEach(s => {
        const li = document.createElement("li");
        li.textContent = s;
        list.appendChild(li);
    });
}

async function refreshAll() {
    try {
        await loadEnergy();
        await Promise.all([loadForecast(), loadFaults(), loadSuggestions()]);
    } catch (err) {
        console.error(err);
        addAlert("Refresh failed: " + err.message, "danger");
    }
}

/* ------------------------------------------------------------------ */
/* Simulation toggle                                                  */
/* ------------------------------------------------------------------ */
function toggleSimulation() {
    const btn = document.getElementById("simulateBtn");
    const status = document.getElementById("simStatus");
    if (simTimer) {
        clearInterval(simTimer);
        simTimer = null;
        btn.textContent = "Start Simulation";
        status.textContent = "Auto-refresh: OFF";
    } else {
        simTimer = setInterval(refreshAll, AUTO_REFRESH_MS);
        btn.textContent = "Stop Simulation";
        status.textContent = `Auto-refresh: every ${AUTO_REFRESH_MS / 1000}s`;
        addAlert("Real-time simulation started.", "info");
    }
}

/* ------------------------------------------------------------------ */
/* Voice assistant                                                    */
/* ------------------------------------------------------------------ */
function speakUsage() {
    const val = document.getElementById("currentValue").textContent;
    const status = document.getElementById("statusBadge").textContent;
    const msg = `Current grid demand is ${val} megawatt hours. Status is ${status}.`;
    if ("speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(msg);
        u.rate = 1;
        u.pitch = 1;
        speechSynthesis.speak(u);
    } else {
        addAlert("Voice synthesis not supported in this browser.", "warn");
    }
}

/* ------------------------------------------------------------------ */
/* Boot                                                               */
/* ------------------------------------------------------------------ */
document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    refreshAll();

    document.getElementById("refreshBtn").addEventListener("click", refreshAll);
    document.getElementById("simulateBtn").addEventListener("click", toggleSimulation);
    document.getElementById("voiceBtn").addEventListener("click", speakUsage);
});
