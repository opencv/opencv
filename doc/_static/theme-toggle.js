(function () {
    var STORAGE_KEY = "opencv-theme";
    var SUN_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
    var MOON_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

    function getTheme() {
        return localStorage.getItem(STORAGE_KEY) ||
               (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        document.documentElement.setAttribute("data-bs-theme", theme);
        localStorage.setItem(STORAGE_KEY, theme);
    }

    function createButton() {
        var btn = document.createElement("button");
        btn.id = "ocv-theme-toggle";
        btn.title = "Toggle light/dark mode";
        btn.setAttribute("aria-label", "Toggle light/dark mode");
        btn.addEventListener("click", function () {
            var current = document.documentElement.getAttribute("data-theme") || getTheme();
            var next = current === "dark" ? "light" : "dark";
            applyTheme(next);
            updateIcon(btn, next);
        });
        return btn;
    }

    function updateIcon(btn, theme) {
        btn.innerHTML = theme === "dark" ? SUN_SVG : MOON_SVG;
    }

    document.addEventListener("DOMContentLoaded", function () {
        var theme = getTheme();
        applyTheme(theme);

        var btn = createButton();
        updateIcon(btn, theme);

        var target = document.querySelector("#navbar-icon-links") ||
                     document.querySelector(".navbar-icon-links") ||
                     document.querySelector(".navbar-persistent--container") ||
                     document.querySelector(".bd-navbar");

        if (target) {
            target.appendChild(btn);
        }
    });
})();
