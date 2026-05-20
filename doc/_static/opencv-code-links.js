(function () {
    "use strict";

    const SKIP_JS = new Set([
        "log","all","add","run","get","set","read","load","save","free","move",
        "call","draw","name","size","type","init","any","has","fit","bind","swap",
        "rows","cols","dims","step","kind","inv","cast","eval","meta","tag","fmt",
        "feed","put","pop","push","top","end","begin","find","next","prev","back",
        "at","do","if","for","new","try","int","std","cv","abs","max","min","exp",
        "pow","sin","cos","tan","len","buf","ptr","ret","err","val","idx","pos",
        "ref","del","hex","oct","dec","num","str","out","in","ok","no","yes","on",
        "off","up","down","left","right","red","blue","cyan","gray","gold","lime",
        "navy","pink","rose","teal","conj","lpi","priv","mov","rvec","warp","what",
        "real","imag","data","rows","step","cols","dims","flags","elem","coef",
        "norm","rank","sort","copy","fill","map","zip","sum","mean","var","std",
        "merge","split","flip","clip","crop","resize","rotate","shift","scale",
        "width","height","depth","channels","empty","total","reshape","clone",
        "row","col","diag","dot","cross","mul","div","mod","neg","cmp","eq","ne",
        "lt","le","gt","ge","and","or","not","xor","count","match","check","test",
    ]);

    function linkifyCode(symbols) {
        // Pygments classes: .n (Name), .nc (Name.Class), .nf (Name.Function),
        // .nv (Name.Variable), .nb (Name.Builtin), .nx (Name.Other),
        // .nl (Name.Label), .na (Name.Attribute), .cpf (Comment.PreprocFile).
        const SELECTOR = "pre .nc, pre .nf, pre .cpf";
        const QUALIFIED_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
        const HEADER_RE = /^["<]?[\w./+-]+\.h(?:pp|h|xx)?[">]?$/;

        document.querySelectorAll(SELECTOR).forEach((el) => {
            // Skip if already linked
            if (el.parentElement && el.parentElement.tagName === "A") return;
            const text = el.textContent.trim();
            if (!text) return;
            if (SKIP_JS.has(text.toLowerCase())) return;
            if (text.length < 4) return;

            // Header file paths inside #include (Pygments class "cpf").
            if (el.classList.contains("cpf") || HEADER_RE.test(text)) {
                const stripped = text.replace(/^["<]+|[">]+$/g, "");
                const fileUrl = symbols[stripped] || symbols[text];
                if (!fileUrl) return;
                const fileLink = document.createElement("a");
                fileLink.href = fileUrl;
                fileLink.target = "_blank";
                fileLink.rel = "noopener";
                fileLink.className = el.className + " opencv-code-link";
                fileLink.title = stripped;
                fileLink.textContent = el.textContent;
                el.replaceWith(fileLink);
                return;
            }

            if (!QUALIFIED_RE.test(text)) return;

            // Try qualified prev::current chain (e.g. cv::Mat)
            let lookupKey = text;
            let prev = el.previousSibling;
            // Skip whitespace text nodes
            while (prev && prev.nodeType === 3 && !prev.textContent.trim()) prev = prev.previousSibling;
            // If preceded by "::", look back further for the namespace token
            if (prev && prev.textContent && /::\s*$/.test(prev.textContent)) {
                let nsNode = prev.previousSibling;
                while (nsNode && nsNode.nodeType === 3 && !nsNode.textContent.trim()) nsNode = nsNode.previousSibling;
                if (nsNode && nsNode.textContent && /^[A-Za-z_][A-Za-z0-9_]*$/.test(nsNode.textContent.trim())) {
                    lookupKey = nsNode.textContent.trim() + "::" + text;
                }
            }

            const url = symbols[lookupKey] || symbols[text];
            if (!url) return;

            const a = document.createElement("a");
            a.href = url;
            a.target = "_blank";
            a.rel = "noopener";
            a.className = el.className + " opencv-code-link";
            a.title = lookupKey;
            a.textContent = el.textContent;
            el.replaceWith(a);
        });
    }

    function symbolMapUrl() {
        if (document.currentScript && document.currentScript.src) {
            return document.currentScript.src.replace(/[^/]+$/, "opencv-symbols.json");
        }
        const scripts = document.querySelectorAll('script[src*="opencv-code-links.js"]');
        if (scripts.length) {
            return scripts[scripts.length - 1].src.replace(/[^/]+$/, "opencv-symbols.json");
        }
        const root = (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.URL_ROOT) || "";
        return root + "_static/opencv-symbols.json";
    }

    function init() {
        fetch(symbolMapUrl())
            .then((r) => (r.ok ? r.json() : {}))
            .then(linkifyCode)
            .catch(() => { /* silently skip if symbol map missing */ });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
