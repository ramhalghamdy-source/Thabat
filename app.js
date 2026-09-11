import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm";

const els = {
  video: document.getElementById("webcam"),
  canvas: document.getElementById("overlay"),
  placeholder: document.getElementById("cameraPlaceholder"),
  startBtn: document.getElementById("startBtn"),
  voiceBtn: document.getElementById("voiceBtn"),
  status: document.getElementById("statusText"),
  badge: document.getElementById("engineBadge"),
  leftShoulder: document.getElementById("leftShoulder"),
  rightShoulder: document.getElementById("rightShoulder"),
  symmetry: document.getElementById("symmetry"),
  trunkLean: document.getElementById("trunkLean"),
  feedbackTitle: document.getElementById("feedbackTitle"),
  feedbackText: document.getElementById("feedbackText"),
  frameQuality: document.getElementById("frameQuality"),
  qualityLabel: document.getElementById("qualityLabel")
};

let poseLandmarker;
let webcamRunning = false;
let lastVideoTime = -1;
let voiceEnabled = true;
let lastSpoken = "";
let lastSpokenAt = 0;
let stream;

// MediaPipe landmark indices.
const P = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24
};

async function initPose() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.55,
      minPosePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55
    });

    els.badge.textContent = "المحرك جاهز";
    els.badge.className = "badge badge-good";
    els.status.textContent = "جاهز. شغّلي الكاميرا وارفعي الذراعين ببطء.";
    els.startBtn.disabled = false;
  } catch (err) {
    console.error(err);
    els.badge.textContent = "تعذر تحميل المحرك";
    els.badge.className = "badge badge-bad";
    els.status.textContent = "تأكدي من اتصال الإنترنت ثم أعيدي تحميل الصفحة.";
  }
}

async function toggleCamera() {
  if (webcamRunning) {
    stopCamera();
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    els.status.textContent = "هذا المتصفح لا يدعم الوصول للكاميرا.";
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 720 }
      },
      audio: false
    });

    els.video.srcObject = stream;
    await els.video.play();
    webcamRunning = true;
    els.placeholder.classList.add("hidden");
    els.startBtn.textContent = "إيقاف الكاميرا";
    els.status.textContent = "اجعلي الرأس والكتفين والذراعين والحوض ظاهرين داخل الإطار.";
    lastVideoTime = -1;
    requestAnimationFrame(predictWebcam);
  } catch (err) {
    console.error(err);
    els.status.textContent = "لم نتمكن من فتح الكاميرا. اسمحي بالوصول للكاميرا واستخدمي HTTPS أو localhost.";
  }
}

function stopCamera() {
  webcamRunning = false;
  stream?.getTracks().forEach(t => t.stop());
  els.video.srcObject = null;
  els.placeholder.classList.remove("hidden");
  els.startBtn.textContent = "تشغيل الكاميرا";
  clearCanvas();
  resetMetrics();
  els.status.textContent = "تم إيقاف الكاميرا.";
}

function clearCanvas() {
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
}

async function predictWebcam() {
  if (!webcamRunning || !poseLandmarker) return;

  resizeCanvasToVideo();

  if (els.video.currentTime !== lastVideoTime) {
    lastVideoTime = els.video.currentTime;
    const result = poseLandmarker.detectForVideo(els.video, performance.now());
    renderResult(result);
  }

  requestAnimationFrame(predictWebcam);
}

function resizeCanvasToVideo() {
  const w = els.video.videoWidth || 960;
  const h = els.video.videoHeight || 720;
  if (els.canvas.width !== w || els.canvas.height !== h) {
    els.canvas.width = w;
    els.canvas.height = h;
  }
}

function renderResult(result) {
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

  if (!result.landmarks?.length) {
    setQuality(0, "غير واضح");
    els.status.textContent = "لم أتعرف على وضعية الجسم بعد. ابتعدي قليلًا وتأكدي من الإضاءة.";
    resetMetrics();
    return;
  }

  const lm = result.landmarks[0];
  const drawingUtils = new DrawingUtils(ctx);
  drawingUtils.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#6ee7f9",
    lineWidth: 4
  });
  drawingUtils.drawLandmarks(lm, {
    color: "#ffffff",
    fillColor: "#173f5f",
    radius: 4
  });

  const required = [
    P.LEFT_SHOULDER, P.RIGHT_SHOULDER,
    P.LEFT_ELBOW, P.RIGHT_ELBOW,
    P.LEFT_WRIST, P.RIGHT_WRIST,
    P.LEFT_HIP, P.RIGHT_HIP
  ];
  const quality = required.reduce((sum, idx) => sum + (lm[idx].visibility ?? 0), 0) / required.length;
  const qualityPct = Math.round(quality * 100);
  setQuality(qualityPct, qualityPct >= 80 ? "ممتاز" : qualityPct >= 60 ? "جيد" : "حسّني الإطار");

  if (qualityPct < 55) {
    els.status.textContent = "بعض المفاصل غير واضحة. أظهري الذراعين والحوض كاملين داخل الصورة.";
    return;
  }

  const leftShoulder = angle(lm[P.LEFT_ELBOW], lm[P.LEFT_SHOULDER], lm[P.LEFT_HIP]);
  const rightShoulder = angle(lm[P.RIGHT_ELBOW], lm[P.RIGHT_SHOULDER], lm[P.RIGHT_HIP]);
  const leftElbow = angle(lm[P.LEFT_SHOULDER], lm[P.LEFT_ELBOW], lm[P.LEFT_WRIST]);
  const rightElbow = angle(lm[P.RIGHT_SHOULDER], lm[P.RIGHT_ELBOW], lm[P.RIGHT_WRIST]);
  const sym = Math.abs(leftShoulder - rightShoulder);
  const lean = trunkLeanDeg(lm);

  els.leftShoulder.textContent = `${Math.round(leftShoulder)}°`;
  els.rightShoulder.textContent = `${Math.round(rightShoulder)}°`;
  els.symmetry.textContent = `${Math.round(sym)}°`;
  els.trunkLean.textContent = `${Math.round(lean)}°`;

  const feedback = makeFeedback({ leftShoulder, rightShoulder, leftElbow, rightElbow, sym, lean });
  els.feedbackTitle.textContent = feedback.title;
  els.feedbackText.textContent = feedback.text;
  els.status.textContent = "التتبع مباشر. ارفعي الذراعين ببطء ثم أعيديهما إلى وضع الراحة.";
  speakFeedback(feedback.voice);
}

function angle(a, b, c) {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const mag1 = Math.hypot(ba.x, ba.y, ba.z);
  const mag2 = Math.hypot(bc.x, bc.y, bc.z);
  if (!mag1 || !mag2) return 0;
  const cos = Math.min(1, Math.max(-1, dot / (mag1 * mag2)));
  return Math.acos(cos) * 180 / Math.PI;
}

function trunkLeanDeg(lm) {
  const shoulderMid = midpoint(lm[P.LEFT_SHOULDER], lm[P.RIGHT_SHOULDER]);
  const hipMid = midpoint(lm[P.LEFT_HIP], lm[P.RIGHT_HIP]);
  const dx = shoulderMid.x - hipMid.x;
  const dy = hipMid.y - shoulderMid.y; // upright body => mostly vertical
  return Math.abs(Math.atan2(dx, Math.max(0.0001, dy)) * 180 / Math.PI);
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function makeFeedback(m) {
  // Prototype thresholds only. Not clinical cutoffs.
  if (m.lean > 12) {
    return {
      title: "ثبتي الجذع",
      text: "لاحظت ميلًا في الجذع أثناء الحركة. حاولي الجلوس باستقامة ثم أعيدي الرفع دون الميل إلى الجانب.",
      voice: "حاولي إبقاء جذعك مستقيمًا."
    };
  }

  if (m.sym > 18) {
    const side = m.leftShoulder < m.rightShoulder ? "اليسرى" : "اليمنى";
    return {
      title: "حسّني التماثل",
      text: `هناك فرق ملحوظ بين الذراعين. ارفعي الذراع ${side} قليلًا حتى تصبح الحركة أكثر تماثلًا.`,
      voice: `ارفعي الذراع ${side} قليلًا.`
    };
  }

  const avgShoulder = (m.leftShoulder + m.rightShoulder) / 2;
  const avgElbow = (m.leftElbow + m.rightElbow) / 2;

  if (avgShoulder >= 120 && avgElbow >= 145 && m.sym <= 12) {
    return {
      title: "ممتاز ✓",
      text: "الحركة متقاربة بين الجانبين والجذع مستقر نسبيًا في هذه المحاولة.",
      voice: "ممتاز. الحركة متوازنة."
    };
  }

  if (avgElbow < 135) {
    return {
      title: "مدّي المرفقين",
      text: "حاولي مد المرفقين أكثر أثناء رفع الذراعين، ضمن المدى المريح لك.",
      voice: "حاولي مد المرفقين قليلًا."
    };
  }

  return {
    title: "استمري في الرفع",
    text: "ارفعي الذراعين تدريجيًا وبشكل مريح مع الحفاظ على الجذع ثابتًا قدر الإمكان.",
    voice: "ارفعي الذراعين تدريجيًا."
  };
}

function setQuality(value, label) {
  els.frameQuality.style.width = `${Math.max(0, Math.min(100, value))}%`;
  els.qualityLabel.textContent = label;
}

function resetMetrics() {
  els.leftShoulder.textContent = "—";
  els.rightShoulder.textContent = "—";
  els.symmetry.textContent = "—";
  els.trunkLean.textContent = "—";
}

function speakFeedback(text) {
  if (!voiceEnabled || !text || !window.speechSynthesis) return;
  const now = performance.now();
  if (text === lastSpoken && now - lastSpokenAt < 4500) return;
  if (now - lastSpokenAt < 2600) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ar-SA";
  utterance.rate = 0.92;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
  lastSpoken = text;
  lastSpokenAt = now;
}

els.startBtn.addEventListener("click", toggleCamera);
els.voiceBtn.addEventListener("click", () => {
  voiceEnabled = !voiceEnabled;
  els.voiceBtn.setAttribute("aria-pressed", String(voiceEnabled));
  els.voiceBtn.textContent = voiceEnabled ? "🔊 التوجيه الصوتي: مفعّل" : "🔇 التوجيه الصوتي: متوقف";
  if (!voiceEnabled) window.speechSynthesis?.cancel();
});

initPose();
