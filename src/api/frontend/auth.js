(function () {
  const API = (window.DERMAI_CONFIG && window.DERMAI_CONFIG.API_URL
    ? window.DERMAI_CONFIG.API_URL
    : "http://127.0.0.1:8000").replace(/\/$/, "");
  const storage = {
    token: "dermai_token",
    username: "dermai_username",
    uid: "dermai_uid",
    email: "dermai_email",
  };

  let signupEmail = "";

  if (localStorage.getItem(storage.token)) {
    window.location.replace("index.html");
    return;
  }

  const tabs = document.querySelectorAll(".tab");
  const loginForm = document.getElementById("loginForm");
  const signupForm = document.getElementById("signupForm");
  const step1 = document.getElementById("step1");
  const step2 = document.getElementById("step2");
  const msg = document.getElementById("msg");
  const otpEmail = document.getElementById("otpEmail");
  const loginBtn = document.getElementById("loginBtn");
  const signupBtn = document.getElementById("signupBtn");
  const verifyBtn = document.getElementById("verifyBtn");
  const resendBtn = document.getElementById("resendBtn");

  function switchTab(tab) {
    tabs.forEach((button) => {
      button.classList.toggle("active", button.dataset.tab === tab);
    });
    loginForm.style.display = tab === "login" ? "flex" : "none";
    signupForm.style.display = tab === "signup" ? "block" : "none";
    clearMsg();
  }

  function showMsg(text, type) {
    msg.textContent = text;
    msg.className = "msg " + type;
    msg.style.display = "block";
  }

  function clearMsg() {
    msg.style.display = "none";
    msg.textContent = "";
    msg.className = "msg";
  }

  function setBusy(button, isBusy, idleText, busyText) {
    button.disabled = isBusy;
    button.textContent = isBusy ? busyText : idleText;
  }

  function saveSession(data) {
    if (!data.access_token) {
      return false;
    }
    localStorage.setItem(storage.token, data.access_token);
    localStorage.setItem(storage.username, data.username || "");
    localStorage.setItem(storage.uid, data.uid || "");
    localStorage.setItem(storage.email, data.email || signupEmail || "");
    return true;
  }

  async function parseJson(res) {
    try {
      return await res.json();
    } catch (error) {
      return {};
    }
  }

  async function doLogin(event) {
    event.preventDefault();
    const email = document.getElementById("loginEmail").value.trim();
    const password = document.getElementById("loginPassword").value;

    if (!email || !password) {
      showMsg("Please fill all login fields.", "error");
      return;
    }

    setBusy(loginBtn, true, "Login", "Logging in...");
    clearMsg();

    try {
      const res = await fetch(API + "/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await parseJson(res);
      if (!res.ok) {
        throw new Error(data.detail || "Login failed.");
      }

      if (!saveSession(data)) {
        throw new Error("Login succeeded but no access token was returned.");
      }

      showMsg("Login successful. Redirecting to the app...", "success");
      window.setTimeout(function () {
        window.location.replace("index.html");
      }, 700);
    } catch (error) {
      showMsg(error.message || "Server not reachable. Is the API running?", "error");
    } finally {
      setBusy(loginBtn, false, "Login", "Logging in...");
    }
  }

  async function doSignup(event) {
    event.preventDefault();
    const username = document.getElementById("suUsername").value.trim();
    const email = document.getElementById("suEmail").value.trim();
    const password = document.getElementById("suPassword").value;
    const gender = document.getElementById("suGender").value;

    if (!username || !email || !password) {
      showMsg("Please fill all signup fields.", "error");
      return;
    }
    if (password.length < 8) {
      showMsg("Password must be at least 8 characters.", "error");
      return;
    }

    setBusy(signupBtn, true, "Create Account & Send OTP", "Creating account...");
    clearMsg();

    try {
      const res = await fetch(API + "/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password, gender }),
      });
      const data = await parseJson(res);
      if (!res.ok) {
        throw new Error(data.detail || "Signup failed.");
      }

      signupEmail = email;
      otpEmail.textContent = email;
      step1.style.display = "none";
      step2.style.display = "block";
      showMsg("Account created. Enter the OTP to finish verification.", "success");
    } catch (error) {
      showMsg(error.message || "Server not reachable.", "error");
    } finally {
      setBusy(signupBtn, false, "Create Account & Send OTP", "Creating account...");
    }
  }

  async function doVerifyOtp(event) {
    event.preventDefault();
    const otp = document.getElementById("otpInput").value.trim();
    if (otp.length !== 6) {
      showMsg("Please enter the 6-digit OTP.", "error");
      return;
    }

    setBusy(verifyBtn, true, "Verify & Continue", "Verifying...");
    clearMsg();

    try {
      const res = await fetch(API + "/auth/verify-otp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: signupEmail, otp: otp }),
      });
      const data = await parseJson(res);
      if (!res.ok) {
        throw new Error(data.detail || "Invalid OTP.");
      }

      if (saveSession(data)) {
        showMsg("Verification complete. Redirecting to the app...", "success");
        window.setTimeout(function () {
          window.location.replace("index.html");
        }, 700);
        return;
      }

      showMsg("Email verified. Please log in to continue.", "success");
      backToLogin();
    } catch (error) {
      showMsg(error.message || "Server not reachable.", "error");
    } finally {
      setBusy(verifyBtn, false, "Verify & Continue", "Verifying...");
    }
  }

  async function resendOtp() {
    if (!signupEmail) {
      showMsg("Sign up first so we know where to send the OTP.", "error");
      return;
    }

    resendBtn.disabled = true;
    try {
      const res = await fetch(API + "/auth/resend-otp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: signupEmail }),
      });
      const data = await parseJson(res);
      if (!res.ok) {
        throw new Error(data.detail || "Unable to resend OTP.");
      }
      showMsg("OTP resent. Check your email.", "success");
    } catch (error) {
      showMsg(error.message || "Server not reachable.", "error");
    } finally {
      window.setTimeout(function () {
        resendBtn.disabled = false;
      }, 30000);
    }
  }

  function backToLogin() {
    step1.style.display = "block";
    step2.style.display = "none";
    document.getElementById("otpInput").value = "";
    switchTab("login");
  }

  tabs.forEach((button) => {
    button.addEventListener("click", function () {
      switchTab(button.dataset.tab);
    });
  });

  loginForm.addEventListener("submit", doLogin);
  document.getElementById("signupDetailsForm").addEventListener("submit", doSignup);
  document.getElementById("otpForm").addEventListener("submit", doVerifyOtp);
  resendBtn.addEventListener("click", resendOtp);
  document.getElementById("backBtn").addEventListener("click", function () {
    step1.style.display = "block";
    step2.style.display = "none";
    clearMsg();
  });

  switchTab("login");
}());