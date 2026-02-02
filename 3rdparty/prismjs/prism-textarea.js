// Textarea syntax highlighting for OpenCV.js tutorials
// This adds Prism.js highlighting to editable code textareas
(function() {
    if (typeof Prism === 'undefined') {
        return;
    }

    function setupTextareaHighlighting(textarea) {
        if (textarea.dataset.highlighted) {
            return;
        }

        var wrapper = document.createElement('div');
        wrapper.className = 'prism-textarea-wrapper';
        wrapper.style.position = 'relative';
        wrapper.style.display = 'inline-block';
        wrapper.style.width = '100%';

        var pre = document.createElement('pre');
        var code = document.createElement('code');
        code.className = 'language-javascript';
        pre.appendChild(code);
        
        var computedStyle = window.getComputedStyle(textarea);
        pre.style.position = 'absolute';
        pre.style.top = '0';
        pre.style.left = '0';
        pre.style.width = '100%';
        pre.style.height = '100%';
        pre.style.margin = '0';
        pre.style.padding = computedStyle.padding;
        pre.style.border = '1px solid transparent';
        pre.style.font = computedStyle.font;
        pre.style.fontSize = computedStyle.fontSize;
        pre.style.lineHeight = computedStyle.lineHeight;
        pre.style.whiteSpace = 'pre-wrap';
        pre.style.wordWrap = 'break-word';
        pre.style.overflow = 'hidden';
        pre.style.pointerEvents = 'none';
        pre.style.background = computedStyle.backgroundColor;
        pre.style.boxSizing = 'border-box';
        
        textarea.style.background = 'transparent';
        textarea.style.position = 'relative';
        textarea.style.zIndex = '2';
        textarea.style.color = 'rgba(0,0,0,0.01)';
        textarea.style.caretColor = '#000';
        textarea.style.MozCaretColor = '#000';
        textarea.style.webkitTextFillColor = 'rgba(0,0,0,0.01)';
        textarea.style.textShadow = '0 0 0 transparent';
        textarea.style.cursor = 'text';
        textarea.style.boxSizing = 'border-box';

        textarea.parentNode.insertBefore(wrapper, textarea);
        wrapper.appendChild(pre);
        wrapper.appendChild(textarea);

        function update() {
            code.textContent = textarea.value;
            Prism.highlightElement(code);
            pre.scrollTop = textarea.scrollTop;
            pre.scrollLeft = textarea.scrollLeft;
        }

        update();

        textarea.addEventListener('input', update);
        textarea.addEventListener('scroll', function() {
            pre.scrollTop = textarea.scrollTop;
            pre.scrollLeft = textarea.scrollLeft;
        });

        textarea.dataset.highlighted = 'true';
    }

    function init() {
        var textareas = document.querySelectorAll('textarea.code');
        for (var i = 0; i < textareas.length; i++) {
            setupTextareaHighlighting(textareas[i]);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
