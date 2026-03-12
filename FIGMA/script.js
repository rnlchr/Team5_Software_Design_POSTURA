document.addEventListener("DOMContentLoaded", () => {
  const loginForm = document.getElementById("login-form");
  const registerForm = document.getElementById("register-form");
  const recordsDeleteButtons = document.querySelectorAll(".records-delete-btn");
  const membersToggle = document.getElementById("members-toggle");
  const membersPanel = document.querySelector(".members-panel");

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
});

