/* Single source of truth for the documentation version dropdown.
 *
 * Consumed by BOTH:
 *   - the legacy Doxygen pages (2.x .. 4.x, 5.0.0-pre): the block below
 *     renders this list into the `#projectnumber` span next to the logo
 *     (needs jQuery, which those pages already load); and
 *   - the new Sphinx (PyData) 5.0 docs: the navbar switcher template reads
 *     `window.OPENCV_DOC_VERSIONS` directly and builds its own <select>.
 *
 * To publish a new release, add ONE entry here ['<label>', '<site-path>']
 * and redeploy this file to the bucket root — the option then appears in
 * the dropdown on every version's pages at once. Do not maintain a second
 * list anywhere. */
window.OPENCV_DOC_VERSIONS = [
    ['4.13.0', '/4.13.0'],
    ['4.12.0', '/4.12.0'],
    ['4.11.0', '/4.11.0'],
    ['5.0', '/5.0'],
    ['5.0.0alpha', '/5.0.0-alpha'],
    ['4.10.0', '/4.10.0'],
    ['4.9.0', '/4.9.0'],
    ['4.8.0', '/4.8.0'],
    ['4.7.0', '/4.7.0'],
    ['4.6.0', '/4.6.0'],
    ['4.5.5', '/4.5.5'],
    ['4.5.4', '/4.5.4'],
    ['4.5.3', '/4.5.3'],
    ['4.5.2', '/4.5.2'],
    ['4.5.1', '/4.5.1'],
    ['4.5.0', '/4.5.0'],
    ['4.4.0', '/4.4.0'],
    ['4.3.0', '/4.3.0'],
    ['4.2.0', '/4.2.0'],
    ['4.1.2', '/4.1.2'],
    ['4.1.1', '/4.1.1'],
    ['4.1.0', '/4.1.0'],
    ['4.0.1', '/4.0.1'],
    ['4.0.0', '/4.0.0'],
    // no more 3.4 releases: ['3.4.21-pre', '/3.4'],
    ['3.4.20-dev', '/3.4'],
    ['3.4.20', '/3.4.20'],
    ['3.4.19', '/3.4.19'],
    ['3.4.18', '/3.4.18'],
    ['3.4.17', '/3.4.17'],
    ['3.4.16', '/3.4.16'],
    ['3.4.15', '/3.4.15'],
    ['3.4.14', '/3.4.14'],
    ['3.4.13', '/3.4.13'],
    ['3.4.12', '/3.4.12'],
    ['3.4.11', '/3.4.11'],
    ['3.4.10', '/3.4.10'],
    ['3.4.9', '/3.4.9'],
    ['3.4.8', '/3.4.8'],
    ['3.4.7', '/3.4.7'],
    ['3.4.6', '/3.4.6'],
    ['3.4.5', '/3.4.5'],
    ['3.4.4', '/3.4.4'],
    ['3.4.3', '/3.4.3'],
    ['3.4.2', '/3.4.2'],
    ['3.4.1', '/3.4.1'],
    ['3.4.0', '/3.4.0'],
    ['3.3.1', '/3.3.1'],
    ['3.3.0', '/3.3.0'],
    ['3.2.0', '/3.2.0'],
    ['3.1.0', '/3.1.0'],
    ['3.0.0', '/3.0.0'],
];

// Present the dropdown newest-first regardless of the order entries were added
// above, so publishing a release stays a one-line append (no manual re-sorting).
// Sort by numeric major.minor.patch descending; suffixes like "-dev"/"-pre"/
// "alpha" carry no digits and are ignored by the key, so same-base variants
// (e.g. "5.0" / "5.0.0-pre" / "5.0.0alpha") tie — Array.sort is stable, so they
// keep the order written above.
window.OPENCV_DOC_VERSIONS.sort(function (a, b) {
    var ka = (a[0].match(/\d+/g) || []).map(Number);
    var kb = (b[0].match(/\d+/g) || []).map(Number);
    for (var i = 0; i < Math.max(ka.length, kb.length); i++) {
        var d = (kb[i] || 0) - (ka[i] || 0);
        if (d) return d;
    }
    return 0;
});

function renderDoxygenVersionDropdown() {
  // Doxygen-only rendering. The Sphinx pages have no `#projectnumber` and
  // no jQuery, so bail early there — they read OPENCV_DOC_VERSIONS above
  // and build their own dropdown in the navbar template.
  if (!document.getElementById("projectnumber") || typeof window.jQuery === "undefined")
      return;
  var versions = window.OPENCV_DOC_VERSIONS;
  var h = '<select>';
  var current_ver = $("#projectnumber")[0].innerText || versions[0][0];
  current_ver = current_ver.trim();
  for (i = 0; i < versions.length; i++) {
      selected = ''
      if(current_ver === versions[i][0])
          selected = ' selected="selected"';
      h += '<option value="' + versions[i][0] + '"' + selected + '>' + versions[i][0] + '</option>';
  }
  h += '</select>';
  $("#projectnumber")[0].innerHTML = h;
  $("#projectnumber select")[0].addEventListener('change', function() {
      var v = $(this).children('option:selected').attr('value');
      var path = undefined;
      for (i = 0; i < versions.length; i++) {
          if(v === versions[i][0]) {
              path = versions[i][1];
              break;
          }
      }
      if (!path) return;
      // Go straight to the chosen version's index via its site-absolute path,
      // so switching works from ANY page of ANY version — no fragile attempt to
      // substitute the current version inside the current URL (which fails on
      // pages whose path isn't "/<version>/...", e.g. the 5.0 C++ API tree).
      // The S3 *website* endpoint serves "/4.13.0/" as index.html, but the plain
      // REST endpoint does not, so append index.html explicitly.
      if (!/\.html?($|[?#])/.test(path))
          path = path.replace(/\/+$/, '') + '/index.html';
      window.location.href = path; // navigate
  });
  return current_ver;
}

// Run as soon as possible. On a normal page load this fires on DOMContentLoaded;
// but on the already-deployed legacy pages this file is pulled in dynamically
// (a loader appended to dynsections.js) and may arrive AFTER DOMContentLoaded
// has fired — in which case render immediately instead of waiting for an event
// that will never come again.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", renderDoxygenVersionDropdown);
} else {
  renderDoxygenVersionDropdown();
}
