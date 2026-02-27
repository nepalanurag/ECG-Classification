/* ECG heartbeat classifier demo. Runs the ONNX model fully in the browser. */
(function () {
  "use strict";

  var CLASS_NAMES = [
    "Normal beat (N)",
    "Supraventricular ectopic (S)",
    "Ventricular ectopic (V)",
    "Fusion beat (F)",
    "Unknown beat (Q)"
  ];
  var N_SAMPLES = 187;
  var TEMPERATURE = 1.74; // fit on held-out beats; see calibration analysis

  var session = null;
  var currentBeat = null; // Float32Array(187)

  var sampleBtns = document.getElementById("sampleBtns");
  var sampleNote = document.getElementById("sampleNote");
  var csvInput = document.getElementById("csvInput");
  var fileInput = document.getElementById("fileInput");
  var clearBtn = document.getElementById("clearBtn");
  var predictBtn = document.getElementById("predictBtn");
  var statusEl = document.getElementById("status");
  var resultEl = document.getElementById("result");
  var plot = document.getElementById("plot");

  function setStatus(msg) { statusEl.textContent = msg; }

  function drawBeat(beat) {
    var ctx = plot.getContext("2d");
    var W = plot.width, H = plot.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#e5e5e5";
    ctx.lineWidth = 1;
    for (var g = 1; g < 4; g++) {
      ctx.beginPath(); ctx.moveTo(0, (H / 4) * g); ctx.lineTo(W, (H / 4) * g); ctx.stroke();
    }
    ctx.strokeStyle = "#b3352b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i < beat.length; i++) {
      var x = (i / (beat.length - 1)) * W;
      var y = H - 12 - beat[i] * (H - 24);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function parseBeat(text) {
    var vals = text.trim().split(/[\s,;]+/).filter(function (s) { return s.length > 0; })
      .map(Number);
    if (vals.length !== N_SAMPLES || vals.some(isNaN)) return null;
    return Float32Array.from(vals);
  }

  // temperature scaling: softmax(log(p) / T); keeps argmax, fixes overconfidence
  function calibrate(probs) {
    var logits = probs.map(function (p) { return Math.log(Math.max(p, 1e-12)); });
    var scaled = logits.map(function (z) { return z / TEMPERATURE; });
    var m = Math.max.apply(null, scaled);
    var exps = scaled.map(function (z) { return Math.exp(z - m); });
    var s = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (e) { return e / s; });
  }

  function showResult(probs, trueName) {
    var cal = calibrate(Array.from(probs));
    var best = 0;
    for (var i = 1; i < cal.length; i++) if (cal[i] > cal[best]) best = i;
    document.getElementById("predLabel").textContent =
      "Predicted: " + CLASS_NAMES[best] + (trueName ? "  (true label: " + trueName + ")" : "");
    document.getElementById("predConf").textContent =
      "Confidence: " + (cal[best] * 100).toFixed(1) + "% (calibrated)";
    var bars = document.getElementById("probBars");
    bars.innerHTML = "";
    for (var k = 0; k < cal.length; k++) {
      var row = document.createElement("div");
      row.className = "bar";
      var lbl = document.createElement("span"); lbl.className = "lbl"; lbl.textContent = CLASS_NAMES[k];
      var track = document.createElement("div"); track.className = "track";
      var fill = document.createElement("div"); fill.className = "fill";
      fill.style.width = (cal[k] * 100).toFixed(1) + "%";
      track.appendChild(fill);
      var val = document.createElement("span"); val.className = "val";
      val.textContent = (cal[k] * 100).toFixed(1) + "%";
      row.appendChild(lbl); row.appendChild(track); row.appendChild(val);
      bars.appendChild(row);
    }
    resultEl.style.display = "block";
  }

  function useBeat(beat, trueName) {
    currentBeat = beat;
    drawBeat(beat);
    resultEl.style.display = "none";
    setStatus(session ? "Beat ready. Press Classify." : "Beat ready. Waiting for model...");
    predictBtn.dataset.trueName = trueName || "";
  }

  // sample beats
  fetch("samples/results.json").then(function (r) { return r.json(); }).then(function (meta) {
    TEMPERATURE = meta.temperature || TEMPERATURE;
    meta.samples.forEach(function (s, i) {
      var b = document.createElement("button");
      b.textContent = s.true_name;
      b.onclick = function () {
        Array.prototype.forEach.call(sampleBtns.children, function (c) { c.classList.remove("active"); });
        b.classList.add("active");
        fetch(s.file).then(function (r) { return r.text(); }).then(function (txt) {
          var beat = parseBeat(txt);
          if (!beat) { setStatus("Could not read sample file."); return; }
          sampleNote.textContent = "Sample: " + s.true_name + ". Model prediction on this beat: " +
            s.predicted_name + " at " + (s.confidence * 100).toFixed(1) + "% calibrated confidence.";
          useBeat(beat, s.true_name);
        });
      };
      sampleBtns.appendChild(b);
      if (i === 0) b.click();
    });
  }).catch(function () { sampleNote.textContent = "Sample beats failed to load."; });

  csvInput.addEventListener("input", function () {
    var beat = parseBeat(csvInput.value);
    if (beat) useBeat(beat, "");
    else if (csvInput.value.trim().length > 0) setStatus("Need exactly 187 comma-separated numbers.");
  });

  fileInput.addEventListener("change", function () {
    var f = fileInput.files[0];
    if (!f) return;
    var rd = new FileReader();
    rd.onload = function () {
      csvInput.value = rd.result;
      var beat = parseBeat(rd.result);
      if (beat) useBeat(beat, "");
      else setStatus("File must contain exactly 187 comma-separated numbers.");
    };
    rd.readAsText(f);
  });

  clearBtn.onclick = function () {
    csvInput.value = ""; fileInput.value = "";
    currentBeat = null; resultEl.style.display = "none";
    plot.getContext("2d").clearRect(0, 0, plot.width, plot.height);
    setStatus("");
  };

  predictBtn.onclick = function () {
    if (!session || !currentBeat) return;
    predictBtn.disabled = true;
    setStatus("Classifying...");
    try {
      var tensor = new ort.Tensor("float32", currentBeat, [1, N_SAMPLES]);
      session.run({ input: tensor }).then(function (out) {
        var probs = out[session.outputNames[0]].data;
        showResult(probs, predictBtn.dataset.trueName);
        setStatus("");
        predictBtn.disabled = false;
      }).catch(function (e) {
        setStatus("Inference failed: " + e.message);
        predictBtn.disabled = false;
      });
    } catch (e) {
      setStatus("Inference failed: " + e.message);
      predictBtn.disabled = false;
    }
  };

  // load the model
  setStatus("Loading model...");
  ort.env.wasm.numThreads = 1;
  ort.InferenceSession.create("ecg_ann.onnx", { executionProviders: ["wasm"] })
    .then(function (s) {
      session = s;
      predictBtn.disabled = false;
      predictBtn.textContent = "Classify beat";
      setStatus(currentBeat ? "Beat ready. Press Classify." : "Model loaded. Pick a sample beat above.");
    })
    .catch(function (e) { setStatus("Could not load model: " + e.message); });
})();
