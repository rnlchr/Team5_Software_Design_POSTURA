document.addEventListener("DOMContentLoaded", () => {
  const loginForm = document.getElementById("login-form");
  const registerForm = document.getElementById("register-form");
  const recordsDeleteButtons = document.querySelectorAll(".records-delete-btn");
  const membersToggle = document.getElementById("members-toggle");
  const membersPanel = document.querySelector(".members-panel");
  const uploadBtn = document.getElementById("upload-btn");
  const posturaFileInput = document.getElementById("postura-file");
  const resultImage = document.getElementById("result-image");
  const resultLabelEl = document.getElementById("result-label");
  const resultConfEl = document.getElementById("result-confidence");
  const resultRecEl = document.getElementById("result-recommendation");

  const validEmailEndings = ["@tip.edu.ph", "@gmail.com", "@yahoo.com"];

  function isValidEmail(email) {
    const lower = (email || "").toLowerCase();
    return validEmailEndings.some((ending) => lower.endsWith(ending));
  }

  if (registerForm) {
    registerForm.addEventListener("submit", (event) => {
      event.preventDefault();

      const usernameInput = registerForm.querySelector('input[name="username"]');
      const emailInput = registerForm.querySelector('input[name="email"]');
      const passwordInput = registerForm.querySelector(
        'input[name="password"]'
      );

      const username = usernameInput ? usernameInput.value.trim() : "";
      const email = emailInput ? emailInput.value.trim() : "";
      const password = passwordInput ? passwordInput.value.trim() : "";

      if (!isValidEmail(email)) {
        alert("Please enter a valid email address.");
        return;
      }

      const user = { username, email, password };
      localStorage.setItem("posturaUser", JSON.stringify(user));

      // After successful registration, go to login page
      window.location.href = "index.html";
    });
  }

  if (loginForm) {
    loginForm.addEventListener("submit", (event) => {
      event.preventDefault();

      const emailInput = loginForm.querySelector('input[name="email"]');
      const email = emailInput ? emailInput.value.trim() : "";

      if (!isValidEmail(email)) {
        alert("Please enter a valid email address.");
        return;
      }

      // For now, skip password validation and go straight to home page
      window.location.href = "home.html";
    });
  }

  recordsDeleteButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const row = button.closest(".records-row");
      if (row) {
        row.remove();
      }
    });
  });

  if (membersToggle && membersPanel) {
    membersToggle.addEventListener("click", () => {
      membersPanel.classList.toggle("is-open");
    });
  }

  function saveHistoryEntry(entry) {
    const key = "posturaHistory";
    const history = JSON.parse(localStorage.getItem(key) || "[]");
    history.unshift(entry);
    if (history.length > 25) history.pop();
    localStorage.setItem(key, JSON.stringify(history));
  }

  if (uploadBtn && posturaFileInput) {
    uploadBtn.addEventListener("click", () => {
      posturaFileInput.click();
    });

    posturaFileInput.addEventListener("change", async () => {
      const file = posturaFileInput.files && posturaFileInput.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = async (e) => {
        const dataUrl = e.target?.result;
        if (typeof dataUrl !== "string") return;

        const formData = new FormData();
        formData.append("image", file);

        try {
          const res = await fetch("http://127.0.0.1:5000/analyze", {
            method: "POST",
            body: formData,
          });

          if (!res.ok) {
            alert("Failed to analyze posture.");
            return;
          }

          const json = await res.json();
          if (json.error) {
            alert(json.error);
            return;
          }

          const latest = {
            imageDataUrl: dataUrl,
            label: json.label,
            confidence: json.confidence,
            recommendation: json.recommendation,
            date: json.date,
          };

          sessionStorage.setItem(
            "posturaLastResult",
            JSON.stringify(latest)
          );
          saveHistoryEntry(latest);

          window.location.href = "results.html";
        } catch (err) {
          alert("Could not reach the analysis server.");
        } finally {
          posturaFileInput.value = "";
        }
      };
      reader.readAsDataURL(file);
    });
  }

  if (resultImage && resultLabelEl && resultConfEl && resultRecEl) {
    const stored = sessionStorage.getItem("posturaLastResult");
    if (stored) {
      const data = JSON.parse(stored);
      resultImage.src = data.imageDataUrl;
      resultLabelEl.textContent = data.label;
      resultConfEl.textContent = (data.confidence * 100).toFixed(1) + "%";
      resultRecEl.textContent = data.recommendation;
    }
  }
});

