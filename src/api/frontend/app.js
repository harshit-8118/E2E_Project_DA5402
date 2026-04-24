(function () {
  const API = (window.DERMAI_CONFIG && window.DERMAI_CONFIG.API_URL
    ? window.DERMAI_CONFIG.API_URL
    : "http://127.0.0.1:8000").replace(/\/$/, "");
  const token = localStorage.getItem("dermai_token");

  if (!token) {
    window.location.replace("auth.html");
    return;
  }

  let currentPredictionId = null;
  let currentFile = null;

  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const results = document.getElementById("results");
  const errorBox = document.getElementById("errorBox");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const logoutBtn = document.getElementById("logoutBtn");
  const profileBtn = document.getElementById("profileBtn");
  const profileModal = document.getElementById("profileModal");

  function authHeaders(extra) {
    const headers = extra || {};
    headers.Authorization = "Bearer " + token;
    return headers;
  }

  function handleUnauthorized() {
    localStorage.removeItem("dermai_token");
    localStorage.removeItem("dermai_username");
    localStorage.removeItem("dermai_uid");
    localStorage.removeItem("dermai_email");
    window.location.replace("auth.html");
  }

  async function checkStatus() {
    const dot = document.getElementById("statusDot");
    const text = document.getElementById("statusText");
    try {
      const res = await fetch(API + "/ready", {
        headers: authHeaders({}),
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        dot.className = "status-dot online";
        text.textContent = "API Ready";
      } else if (res.status === 401 || res.status === 403) {
        handleUnauthorized();
      } else {
        dot.className = "status-dot";
        text.textContent = "Model Loading...";
      }
    } catch (error) {
      dot.className = "status-dot";
      text.textContent = "API Offline";
    }
  }

  function show(id, display) {
    document.getElementById(id).style.display = display || "block";
  }

  function hide(id) {
    document.getElementById(id).style.display = "none";
  }

  function spinMsg(text) {
    document.getElementById("spinnerMsg").textContent = text;
  }

  function showError(message) {
    errorBox.textContent = "Error: " + message;
    errorBox.style.display = "block";
  }

  async function parseJson(res) {
    try {
      return await res.json();
    } catch (error) {
      return {};
    }
  }

  function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("previewImg").src = event.target.result;
      document.getElementById("origImg").src = event.target.result;
    };
    reader.readAsDataURL(file);

    document.getElementById("fileName").textContent = file.name;
    document.getElementById("fileSize").textContent =
      (file.size / 1024).toFixed(1) + " KB - " + file.type;

    show("previewWrap", "flex");
    hide("dropZone");
    hide("results");
    hide("errorBox");
  }

  async function runAnalysis() {
    if (!currentFile) {
      showError("Please choose an image first.");
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.querySelector("span").textContent = "Analyzing...";
    hide("results");
    hide("errorBox");
    show("spinnerWrap", "flex");
    spinMsg("Preprocessing image...");

    try {
      const formData = new FormData();
      formData.append("file", currentFile);

      spinMsg("Running inference...");
      const res = await fetch(API + "/predict", {
        method: "POST",
        headers: authHeaders({}),
        body: formData,
      });
      if (res.status === 401 || res.status === 403) {
        handleUnauthorized();
        return;
      }
      if (!res.ok) {
        const err = await res.json().catch(function () {
          return { detail: res.statusText };
        });
        throw new Error(err.detail || "Prediction failed.");
      }

      spinMsg("Generating Grad-CAM...");
      const data = await res.json();
      renderResults(data);
    } catch (error) {
      showError(error.message || "An unexpected error occurred.");
    } finally {
      hide("spinnerWrap");
      analyzeBtn.disabled = false;
      analyzeBtn.querySelector("span").textContent = "Analyze Image";
    }
  }

  function renderResults(data) {
    currentPredictionId = data.prediction_id;
    document.getElementById("gradcamImg").src = data.gradcam_image || "";
    document.getElementById("predClass").textContent = data.display_name;

    const riskBadge = document.getElementById("riskBadge");
    riskBadge.className = "risk-badge risk-" + data.risk_level;
    riskBadge.textContent = data.risk_level + " Risk";

    const scores = Object.entries(data.all_scores).sort(function (a, b) {
      return b[1] - a[1];
    });
    const topClass = data.predicted_class;
    document.getElementById("confBars").innerHTML = scores.map(function (entry) {
      const cls = entry[0];
      const val = entry[1];
      return (
        '<div class="conf-row">' +
          '<span class="cls-name">' + cls + "</span>" +
          '<div class="conf-track">' +
            '<div class="conf-fill ' + (cls === topClass ? "top" : "") +
            '" style="width:' + Math.round(val * 100) + '%"></div>' +
          "</div>" +
          '<span class="conf-val">' + (val * 100).toFixed(1) + "%</span>" +
        "</div>"
      );
    }).join("");

    document.getElementById("inferTime").textContent =
      "Inference: " + data.inference_ms + "ms - Total: " + data.total_ms + "ms";
    document.getElementById("symptoms").innerHTML = data.symptoms.map(function (symptom) {
      return "<li>" + symptom + "</li>";
    }).join("");
    document.getElementById("advisory").textContent = data.advisory;
    document.getElementById("sources").innerHTML = data.sources.map(function (source) {
      return '<a href="' + source.url + '" target="_blank" rel="noreferrer" class="source-link">' +
        source.name + "</a>";
    }).join("");
    document.getElementById("disclaimer").textContent = data.disclaimer;

    document.getElementById("thumbUp").className = "btn-feedback";
    document.getElementById("thumbDown").className = "btn-feedback";
    document.getElementById("thumbUp").disabled = false;
    document.getElementById("thumbDown").disabled = false;
    document.getElementById("feedbackComment").value = "";
    hide("feedbackThanks");

    show("results", "block");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function submitFeedback(vote) {
    if (!currentPredictionId) {
      return;
    }
    const comment = document.getElementById("feedbackComment").value.trim();

    document.getElementById("thumbUp").disabled = true;
    document.getElementById("thumbDown").disabled = true;
    if (vote === "thumbs_up") {
      document.getElementById("thumbUp").classList.add("active-up");
    } else {
      document.getElementById("thumbDown").classList.add("active-down");
    }

    try {
      const res = await fetch(API + "/feedback", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          prediction_id: currentPredictionId,
          vote: vote,
          comment: comment,
        }),
      });
      if (res.status === 401 || res.status === 403) {
        handleUnauthorized();
        return;
      }
      if (!res.ok) {
        const data = await parseJson(res);
        throw new Error(data.detail || "Unable to save feedback.");
      }
      show("feedbackThanks", "flex");
    } catch (error) {
      document.getElementById("thumbUp").disabled = false;
      document.getElementById("thumbDown").disabled = false;
      showError(error.message || "Unable to save feedback.");
      return;
    }
  }

  function formatTimestamp(value) {
    if (!value) {
      return "-";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value;
    }
    return date.toLocaleString();
  }

  function renderPredictionHistory(predictions) {
    const historyEl = document.getElementById("predictionHistory");
    if (!predictions || predictions.length === 0) {
      historyEl.innerHTML = '<p class="history-empty">No predictions available yet.</p>';
      return;
    }

    historyEl.innerHTML = predictions.map(function (prediction) {
      return (
        '<article class="history-item">' +
          '<div class="history-item-head">' +
            '<strong>' + prediction.predicted_class + '</strong>' +
            '<span class="risk-badge risk-' + prediction.risk_level + '">' + prediction.risk_level + " Risk</span>" +
          "</div>" +
          '<div class="history-item-meta">Confidence: ' + (Number(prediction.confidence || 0) * 100).toFixed(1) +
          "% - " + formatTimestamp(prediction.timestamp) + "</div>" +
        "</article>"
      );
    }).join("");
  }

  async function openProfileModal() {
    try {
      const responses = await Promise.all([
        fetch(API + "/auth/me", { headers: authHeaders({}) }),
        fetch(API + "/auth/my-predictions?limit=10", { headers: authHeaders({}) }),
      ]);

      for (let index = 0; index < responses.length; index += 1) {
        if (responses[index].status === 401 || responses[index].status === 403) {
          handleUnauthorized();
          return;
        }
      }

      const meRes = responses[0];
      const predictionsRes = responses[1];
      const meData = await parseJson(meRes);
      const predictionData = await parseJson(predictionsRes);

      if (!meRes.ok) {
        throw new Error(meData.detail || "Unable to load profile.");
      }
      if (!predictionsRes.ok) {
        throw new Error(predictionData.detail || "Unable to load prediction history.");
      }

      document.getElementById("profileUsername").textContent = meData.username || "-";
      document.getElementById("profileEmail").textContent = meData.email || "-";
      document.getElementById("profileGender").textContent = meData.gender || "-";
      document.getElementById("profileVerified").textContent = meData.verified ? "Yes" : "No";
      document.getElementById("profilePredictionCount").textContent = meData.prediction_count || 0;
      document.getElementById("profileFeedbackCount").textContent = meData.feedback_count || 0;
      renderPredictionHistory(predictionData.predictions || []);

      profileModal.style.display = "flex";
    } catch (error) {
      showError(error.message || "Unable to load profile.");
    }
  }

  function closeProfileModal() {
    profileModal.style.display = "none";
  }

  function resetAll() {
    currentFile = null;
    currentPredictionId = null;
    fileInput.value = "";
    show("dropZone", "flex");
    hide("previewWrap");
    hide("spinnerWrap");
    hide("results");
    hide("errorBox");
  }

  dropZone.addEventListener("dragover", function (event) {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("drag-over");
  });
  dropZone.addEventListener("drop", function (event) {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleFile(file);
    }
  });
  dropZone.addEventListener("click", function () {
    fileInput.click();
  });
  document.getElementById("browseTrigger").addEventListener("click", function (event) {
    event.stopPropagation();
    fileInput.click();
  });
  fileInput.addEventListener("change", function (event) {
    if (event.target.files[0]) {
      handleFile(event.target.files[0]);
    }
  });
  analyzeBtn.addEventListener("click", runAnalysis);
  document.getElementById("clearBtn").addEventListener("click", resetAll);
  document.getElementById("resetBtn").addEventListener("click", resetAll);
  document.getElementById("thumbUp").addEventListener("click", function () {
    submitFeedback("thumbs_up");
  });
  document.getElementById("thumbDown").addEventListener("click", function () {
    submitFeedback("thumbs_down");
  });
  profileBtn.addEventListener("click", openProfileModal);
  document.getElementById("profileModalClose").addEventListener("click", closeProfileModal);
  profileModal.addEventListener("click", function (event) {
    if (event.target === profileModal) {
      closeProfileModal();
    }
  });
  logoutBtn.addEventListener("click", handleUnauthorized);

  hide("previewWrap");
  hide("spinnerWrap");
  hide("results");
  hide("errorBox");
  checkStatus();
  window.setInterval(checkStatus, 15000);
}());
