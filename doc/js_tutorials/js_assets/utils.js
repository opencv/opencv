function loadImageToCanvas(url, cavansId) {
  var canvas = document.getElementById(cavansId);
  var ctx = canvas.getContext('2d');
  var img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = function() {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0, img.width, img.height);
  }
  img.src = url;
}
