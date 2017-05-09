function toggleVisibility(linkObj)
{
 var base = $(linkObj).attr('id');
 var summary = $('#'+base+'-summary');
 var content = $('#'+base+'-content');
 var trigger = $('#'+base+'-trigger');
 var src=$(trigger).attr('src');
 if (content.is(':visible')===true) {
   content.hide();
   summary.show();
   $(linkObj).addClass('closed').removeClass('opened');
   $(trigger).attr('src',src.substring(0,src.length-8)+'closed.png');
 } else {
   content.show();
   summary.hide();
   $(linkObj).removeClass('closed').addClass('opened');
   $(trigger).attr('src',src.substring(0,src.length-10)+'open.png');
 } 
 return false;
}

function updateStripes()
{
  $('table.directory tr').
       removeClass('even').filter(':visible:even').addClass('even');
}
function toggleLevel(level)
{
  $('table.directory tr').each(function(){ 
    var l = this.id.split('_').length-1;
    var i = $('#img'+this.id.substring(3));
    var a = $('#arr'+this.id.substring(3));
    if (l<level+1) {
      i.attr('src','ftv2folderopen.png');
      a.attr('src','ftv2mnode.png');
      $(this).show();
    } else if (l==level+1) {
      i.attr('src','ftv2folderclosed.png');
      a.attr('src','ftv2pnode.png');
      $(this).show();
    } else {
      $(this).hide();
    }
  });
  updateStripes();
}
function toggleFolder(id) 
{
  var n = $('[id^=row_'+id+']');
  var i = $('[id^=img_'+id+']');
  var a = $('[id^=arr_'+id+']');
  var c = n.slice(1);
  if (c.filter(':first').is(':visible')===true) {
    i.attr('src','ftv2folderclosed.png');
    a.attr('src','ftv2pnode.png');
    c.hide();
  } else {
    i.attr('src','ftv2folderopen.png');
    a.attr('src','ftv2mnode.png');
    c.show();
  }
  updateStripes();
}

function toggleInherit(id)
{
  var rows = $('tr.inherit.'+id);
  var img = $('tr.inherit_header.'+id+' img');
  var src = $(img).attr('src');
  if (rows.filter(':first').is(':visible')===true) {
    rows.css('display','none');
    $(img).attr('src',src.substring(0,src.length-8)+'closed.png');
  } else {
    rows.css('display','table-row'); // using show() causes jump in firefox
    $(img).attr('src',src.substring(0,src.length-10)+'open.png');
  }
}

