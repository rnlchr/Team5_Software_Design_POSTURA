document.addEventListener("DOMContentLoaded", () => {
  const loginForm           = document.getElementById("login-form");
  const registerForm        = document.getElementById("register-form");
  const recordsDeleteButtons = document.querySelectorAll(".records-delete-btn");
  const membersToggle       = document.getElementById("members-toggle");
  const membersPanel        = document.querySelector(".members-panel");
  const uploadBtn           = document.getElementById("upload-btn");
  const posturaFileInput    = document.getElementById("postura-file");
  const resultImage         = document.getElementById("result-image");
  const resultLabelEl       = document.getElementById("result-label");
  const resultConfEl        = document.getElementById("result-confidence");
  const resultRecEl         = document.getElementById("result-recommendation");
  const recordsBody         = document.getElementById("records-body");
  const recordsEmpty        = document.getElementById("records-empty-msg");
  const pageLabel           = document.getElementById("page-label");
  const prevPageBtn         = document.getElementById("prev-page");
  const nextPageBtn         = document.getElementById("next-page");

  const API = "http://127.0.0.1:5000";

  // ── Register ─────────────────────────────────────────────────────
  if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = registerForm.querySelector('input[name="username"]').value.trim();
      const email    = registerForm.querySelector('input[name="email"]').value.trim();
      const password = registerForm.querySelector('input[name="password"]').value.trim();

      try {
        const res = await fetch(`${API}/api/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, email, password }),
        });
        const json = await res.json();
        if (!res.ok) {
          alert(json.error || "Registration failed.");
          return;
        }
        alert("Account created! Please log in.");
        window.location.href = "index.html";
      } catch {
        alert("Could not reach server.");
      }
    });
  }

  // ── Login ─────────────────────────────────────────────────────────
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const email    = loginForm.querySelector('input[name="email"]').value.trim();
      const password = loginForm.querySelector('input[name="password"]').value.trim();

      try {
        const res = await fetch(`${API}/api/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ email, password }),
        });
        const json = await res.json();
        if (!res.ok) {
          alert(json.error || "Login failed.");
          return;
        }
        sessionStorage.setItem("posturaUser", JSON.stringify({
          username: json.username,
          email:    json.email,
        }));
        window.location.href = "home.html";
      } catch {
        alert("Could not reach server.");
      }
    });
  }

  // ── Members panel toggle ──────────────────────────────────────────
  if (membersToggle && membersPanel) {
    membersToggle.addEventListener("click", () => {
      membersPanel.classList.toggle("is-open");
    });
  }

  // ── Upload + Analyze ──────────────────────────────────────────────
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
          const res = await fetch(`${API}/analyze`, {
            method: "POST",
            credentials: "include",
            body: formData,
          });

          if (res.status === 401) {
            alert("Please log in first.");
            window.location.href = "index.html";
            return;
          }

          if (!res.ok) {
            const json = await res.json();
            alert(json.error || "Failed to analyze posture.");
            return;
          }

          const json = await res.json();
          if (json.error) {
            alert(json.error);
            return;
          }

          sessionStorage.setItem("posturaLastResult", JSON.stringify({
            imageDataUrl:   dataUrl,
            label:          json.label,
            confidence:     json.confidence,
            recommendation: json.recommendation,
            date:           json.date,
          }));

          window.location.href = "results.html";
        } catch {
          alert("Could not reach the analysis server.");
        } finally {
          posturaFileInput.value = "";
        }
      };
      reader.readAsDataURL(file);
    });
  }

  // ── Results page ──────────────────────────────────────────────────
  if (resultImage && resultLabelEl && resultConfEl && resultRecEl) {
    const stored = sessionStorage.getItem("posturaLastResult");
    if (stored) {
      const data = JSON.parse(stored);
      resultImage.src          = data.imageDataUrl;
      resultLabelEl.textContent = data.label.toUpperCase();
      resultConfEl.textContent  = (data.confidence * 100).toFixed(1) + "%";
      resultRecEl.textContent   = data.recommendation;

      // Color label green/red
      resultLabelEl.style.color = data.label === "good" ? "#2e7d32" : "#c62828";
      resultLabelEl.style.fontWeight = "600";
    }
  }

  // ── Records page ──────────────────────────────────────────────────
  let currentPage = 1;

  async function loadRecords(page) {
    if (!recordsBody) return;

    try {
      const res = await fetch(`${API}/api/records?page=${page}`, {
        credentials: "include",
      });

      if (res.status === 401) {
        window.location.href = "index.html";
        return;
      }

      const json = await res.json();
      currentPage = json.page;

      // Update page label
      if (pageLabel) {
        pageLabel.textContent = `Page ${json.page} of ${json.total_pages}`;
      }

      // Prev/Next buttons
      if (prevPageBtn) prevPageBtn.style.display = json.page <= 1 ? "none" : "inline";
      if (nextPageBtn) nextPageBtn.style.display = json.page >= json.total_pages ? "none" : "inline";

      // Clear existing rows
      recordsBody.innerHTML = "";

      if (json.records.length === 0) {
        if (recordsEmpty) recordsEmpty.style.display = "block";
        return;
      }

      if (recordsEmpty) recordsEmpty.style.display = "none";

      json.records.forEach((record) => {
        const row = document.createElement("div");
        row.className = "records-row";
        row.dataset.id = record.id;

        const labelColor = record.label === "good" ? "#2e7d32" : "#c62828";

        row.innerHTML = `
          <div class="records-cell">
            <div class="records-thumb"></div>
          </div>
          <div class="records-cell" style="color: ${labelColor}; font-weight: 600;">
            ${record.label.toUpperCase()}
          </div>
          <div class="records-cell">${record.recommendation}</div>
          <div class="records-cell">${record.date}</div>
          <div class="records-cell">
            <button type="button" class="records-delete-btn" data-id="${record.id}">Delete</button>
          </div>
        `;

        recordsBody.appendChild(row);

        // Delete button
        row.querySelector(".records-delete-btn").addEventListener("click", async () => {
          if (!confirm("Delete this record?")) return;
          try {
            const delRes = await fetch(`${API}/api/records/${record.id}`, {
              method: "DELETE",
              credentials: "include",
            });
            if (delRes.ok) {
              row.remove();
              // Reload if page is now empty
              const remaining = recordsBody.querySelectorAll(".records-row").length;
              if (remaining === 0) loadRecords(Math.max(1, currentPage - 1));
            }
          } catch {
            alert("Could not delete record.");
          }
        });
      });

    } catch {
      if (recordsEmpty) {
        recordsEmpty.textContent = "Could not load records. Make sure the server is running.";
        recordsEmpty.style.display = "block";
      }
    }
  }

  // Pagination buttons
  if (prevPageBtn) {
    prevPageBtn.addEventListener("click", () => loadRecords(currentPage - 1));
  }
  if (nextPageBtn) {
    nextPageBtn.addEventListener("click", () => loadRecords(currentPage + 1));
  }

  // Load records on page load
  if (recordsBody) {
    loadRecords(1);
  }
});