function insertIframe (elementId, iframeSrc)
{
  var iframe;
  if (document.createElement && (iframe = document.createElement('iframe')))
  {
    iframe.src = unescape(iframeSrc);
    iframe.width = "100%";
    iframe.height = "511px";
    var element = document.getElementById(elementId);
    element.parentNode.replaceChild(iframe, element);
  }
}
