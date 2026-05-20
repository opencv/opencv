(function () {
    var COPY_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    var CHECK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

    var HOME_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>';

    function buildBreadcrumbs() {
        var path = window.location.pathname;
        if (path === "/" || path === "/index.html" || path === "/index.htm") return;

        var crumbs = [];
        document.querySelectorAll("#bd-docs-nav li.current").forEach(function (li) {
            var a = li.querySelector(":scope > a");
            if (a) crumbs.push({ text: a.textContent.trim(), href: a.getAttribute("href") });
        });

        var rootHref = (document.querySelector(".navbar-brand") || {}).getAttribute
            ? document.querySelector(".navbar-brand").getAttribute("href") : "/";

        var html = '<nav class="ocv-breadcrumb" aria-label="breadcrumb">'
            + '<a href="' + rootHref + '" class="ocv-breadcrumb-home" title="Home">' + HOME_SVG + '</a>';

        if (crumbs.length > 0) {
            crumbs.forEach(function (c, i) {
                html += '<span class="ocv-breadcrumb-sep">›</span>';
                if (i < crumbs.length - 1) {
                    html += '<a href="' + c.href + '">' + c.text + '</a>';
                } else {
                    html += '<span class="ocv-breadcrumb-current">' + c.text + '</span>';
                }
            });
        } else {
            var h1El = document.querySelector(".bd-content h1, article.bd-article h1");
            if (h1El) {
                var title = h1El.textContent.trim().replace(/\s*¶\s*$/, "");
                if (title) html += '<span class="ocv-breadcrumb-sep">›</span>'
                    + '<span class="ocv-breadcrumb-current">' + title + '</span>';
            }
        }

        html += '</nav>';

        var h1 = document.querySelector(".bd-content h1, article.bd-article h1");
        if (h1) h1.insertAdjacentHTML("beforebegin", html);
    }

    function initTocHighlight() {
        var tocLinks = document.querySelectorAll("#bd-toc-nav a");
        if (!tocLinks.length) return;
        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    var id = entry.target.getAttribute("id");
                    tocLinks.forEach(function (a) {
                        a.classList.toggle("ocv-toc-active", a.getAttribute("href") === "#" + id);
                    });
                }
            });
        }, { rootMargin: "-80px 0px -70% 0px" });
        document.querySelectorAll("section[id], div[id].section").forEach(function (s) {
            observer.observe(s);
        });
    }

    function initBackToTop() {
        var btn = document.createElement("button");
        btn.id = "ocv-back-to-top";
        btn.title = "Back to top";
        btn.setAttribute("aria-label", "Back to top");
        btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg> Back to top';
        document.body.appendChild(btn);

        window.addEventListener("scroll", function () {
            btn.classList.toggle("ocv-btt-visible", window.scrollY > 150);
        }, { passive: true });
        /* also check on load in case page is already scrolled */
        if (window.scrollY > 150) btn.classList.add("ocv-btt-visible");

        btn.addEventListener("click", function () {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        buildBreadcrumbs();
        initTocHighlight();
        initBackToTop();

        var sidebar = document.querySelector(".bd-sidebar");
        if (sidebar) {
            sidebar.style.setProperty("padding-left", "0", "important");
            sidebar.style.setProperty("padding-right", "0", "important");
        }
        var containerXl = document.querySelector(".container-xl > .row");
        if (containerXl) {
            containerXl.style.setProperty("margin-left", "0", "important");
            containerXl.style.setProperty("margin-right", "0", "important");
            var parentContainer = containerXl.parentElement;
            if (parentContainer) {
                parentContainer.style.setProperty("padding-left", "0", "important");
                parentContainer.style.setProperty("padding-right", "0", "important");
            }
        }
        document.querySelectorAll(".bd-content ol > li > p:first-child > strong:first-child").forEach(function (el) {
            if (el.textContent.endsWith(".")) {
                el.textContent = el.textContent.slice(0, -1);
            }
        });

        document.querySelectorAll("div.highlight").forEach(function (block) {
            var btn = document.createElement("button");
            btn.className = "ocv-copy-btn";
            btn.title = "Copy";
            btn.setAttribute("aria-label", "Copy code");
            btn.innerHTML = COPY_SVG;

            btn.addEventListener("click", function () {
                var pre = block.querySelector("pre");
                var text = pre ? pre.innerText : "";
                function onCopied() {
                    btn.innerHTML = CHECK_SVG;
                    btn.classList.add("ocv-copy-success");
                    setTimeout(function () {
                        btn.innerHTML = COPY_SVG;
                        btn.classList.remove("ocv-copy-success");
                    }, 1500);
                }
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(text).then(onCopied);
                } else {
                    var ta = document.createElement("textarea");
                    ta.value = text;
                    ta.style.position = "fixed";
                    ta.style.opacity = "0";
                    document.body.appendChild(ta);
                    ta.focus();
                    ta.select();
                    try { document.execCommand("copy"); onCopied(); } catch (e) {}
                    document.body.removeChild(ta);
                }
            });

            block.appendChild(btn);
        });
    });
})();

function ocvTabClick(btn) {
    var sync = btn.dataset.sync;
    var tabset = btn.closest(".ocv-tabset");
    var labels = Array.from(tabset.querySelectorAll(".ocv-tab-btn"));
    var panels = Array.from(tabset.querySelectorAll(".ocv-tab-panel"));
    var idx = labels.indexOf(btn);

    labels.forEach(function (b) { b.classList.remove("ocv-tab-active"); });
    panels.forEach(function (p) { p.classList.remove("ocv-tab-active"); });
    btn.classList.add("ocv-tab-active");
    if (panels[idx]) panels[idx].classList.add("ocv-tab-active");

    if (sync) {
        document.querySelectorAll(".ocv-tabset").forEach(function (ts) {
            if (ts === tabset) return;
            var syncBtn = Array.from(ts.querySelectorAll(".ocv-tab-btn")).find(function (b) {
                return b.dataset.sync === sync;
            });
            if (syncBtn && !syncBtn.classList.contains("ocv-tab-active")) {
                ocvTabClick(syncBtn);
            }
        });
    }
}
