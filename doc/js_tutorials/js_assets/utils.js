function loadImageToCanvas(url, cavansId) { // eslint-disable-line no-unused-vars
  let canvas = document.getElementById(cavansId);
  let ctx = canvas.getContext('2d');
  let img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = function() {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0, img.width, img.height);
  };
  img.src = url;
}

function executeCode(codeEditorId, errorOutputId) { // eslint-disable-line no-unused-vars
  let code = document.getElementById(codeEditorId).value;
  try {
    eval(code);
    document.getElementById(errorOutputId).innerHTML = ' ';
  } catch (err) {
    document.getElementById(errorOutputId).innerHTML = err;
  }
}

function loadCode(scriptId, codeEditorId) { // eslint-disable-line no-unused-vars
  let scriptNode = document.getElementById(scriptId);
  let codeEditor = document.getElementById(codeEditorId);
  if (scriptNode.type !== 'text/code-snippet') {
    throw Error('Unknown code snippet type');
  }
  codeEditor.value = scriptNode.text.replace(/^\n/, '');
}
